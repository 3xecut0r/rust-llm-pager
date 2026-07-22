from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

import torch
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from ..batched_decode import batched_decode_step
from ..paged_model import PagedModel

logger = logging.getLogger(__name__)


class SessionCapacityError(RuntimeError):
    """Raised by create_session() when the persistent-session registry is already full."""


class SessionBusyError(RuntimeError):
    """Raised when a session already has an in-flight turn (submit_turn) or delete_session is asked to
    remove one -- a session serializes its own turns, one at a time, same as PagedModel.generate() itself."""


class SchedulerAtCapacityError(RuntimeError):
    """Raised by submit()/submit_turn() when the incoming queue is already at max_queue_depth -- an
    explicit backpressure signal (mapped to HTTP 429 by app.py) instead of queuing indefinitely."""


@dataclass
class SessionState:
    """
    A multi-turn conversation: one PagedModel kept alive across several
    /v1/completions calls, addressed by a server-generated session_id.
    input_ids/attention_mask are None until the session's first turn, then
    hold the full growing sequence (this turn's input + every token
    generated so far) -- exactly what PagedModel._extend_session expects on
    the next turn, no separate bookkeeping needed beyond copying it over in
    _retire() once a turn finishes.
    """

    session_id: str
    paged_model: PagedModel
    input_ids: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


@dataclass
class ServingRequest:
    # One HTTP request's worth of state: its own PagedModel session, its
    # growing input_ids/attention_mask (extended by one token per round-robin
    # turn), and the plumbing an HTTP handler needs to stream tokens back or
    # wait for completion without polling.
    request_id: str
    paged_model: PagedModel
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    max_new_tokens: int
    eos_token_id: int | None
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    session_id: str | None = None
    generated_tokens: list[int] = field(default_factory=list)
    output_queue: "queue.Queue[int | None]" = field(default_factory=queue.Queue)
    done: threading.Event = field(default_factory=threading.Event)
    error: Exception | None = None


class ContinuousBatchingScheduler:
    """
    Decode scheduler for many independent PagedModel sessions: instead of
    running each request's full generation to completion before starting the
    next, every session advances by one decode step per round.

    Stage 2 (default when use_streaming_attention=True): sessions that have
    already taken their first step and currently share the same tail length
    are advanced together via batched_decode_step -- one combined
    model.forward() call for the whole group instead of one call per
    session, real vLLM-style batched continuous batching, not just
    interleaving. A brand-new request's first step (prefill + first decode
    token) always goes through PagedModel.generate() directly -- batching
    prefills of different-length prompts together is a different, harder
    problem batched_decode_step doesn't attempt -- and any session that
    doesn't share a tail-length group with at least one other active session
    this round falls back to its own generate() call too, exactly like
    Stage 1's plain round-robin (no session is ever blocked waiting for a
    batching partner that isn't there). Falls back to pure round-robin
    entirely when use_streaming_attention=False, since batched_decode_step
    only supports the streaming path.

    Each request gets its own PagedModel instance (cheap to construct --
    PagedModel.__init__ does no GPU work until the first generate() call),
    constructed with an equal share of a total VRAM/RAM budget across
    max_concurrent_sessions. This is a conservative first cut: PagedModel
    instances don't coordinate GPU memory with each other, so the safe bound
    is (per-session budget) * max_concurrent_sessions <= real headroom.

    Multi-turn conversations: create_session()/submit_turn()/delete_session()
    let a client keep one PagedModel (and its KV cache) alive across several
    /v1/completions calls instead of the one-shot submit() path starting a
    fresh session every time. A persistent session and a one-shot submit()
    call both draw from the same max_concurrent_sessions-sized pool of
    PagedModel instances, but -- in this first pass -- each is capped
    independently (len(_sessions) vs len(_active)/_incoming), not by one
    combined counter. A client holding many persistent sessions open while
    also issuing many one-shot completions at once can therefore exceed the
    per-session VRAM/RAM budget's original assumption; a known, disclosed
    gap, not silently ignored. There is also no idle-timeout eviction for
    persistent sessions -- cleanup is entirely client-driven via
    delete_session()/DELETE /v1/sessions/{id}.
    """

    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_concurrent_sessions: int,
        total_vram_budget: int,
        total_ram_budget: int,
        policy: str = "sinks_heavy_hitter",
        tokens_per_block: int = 16,
        use_streaming_attention: bool = True,
        streaming_group_size_blocks: int = 64,
        max_queue_depth: int = 100,
        device: torch.device | str = "cpu",
        metrics_registry: CollectorRegistry | None = None,
        device_label: str = "default",
    ):
        if max_concurrent_sessions < 1:
            raise ValueError(f"max_concurrent_sessions must be >= 1, got {max_concurrent_sessions}")
        if max_queue_depth < 0:
            raise ValueError(f"max_queue_depth must be >= 0, got {max_queue_depth}")

        self._model = model
        self._tokenizer = tokenizer
        self._max_concurrent_sessions = max_concurrent_sessions
        self._max_queue_depth = max_queue_depth
        self._vram_budget_per_session = total_vram_budget // max_concurrent_sessions
        self._ram_budget_per_session = total_ram_budget // max_concurrent_sessions
        self._paged_model_kwargs = dict(
            policy=policy,
            tokens_per_block=tokens_per_block,
            use_streaming_attention=use_streaming_attention,
            streaming_group_size_blocks=streaming_group_size_blocks,
        )
        # batched_decode_step only supports the streaming path -- with it off,
        # every step falls back to the plain per-session generate() call.
        self._can_batch = use_streaming_attention

        # Explicit, not introspected from the model (next(model.parameters()).device
        # would break every CPU-only test that constructs this with model=object()).
        # submit()/submit_turn() move incoming tensors here -- the HTTP layer no
        # longer needs to know which device a request will end up on, which matters
        # once a MultiGpuScheduler router picks the destination scheduler *after*
        # tokenization (see pager_hf/serving/multi_gpu.py).
        self._device = torch.device(device)
        self._device_label = device_label

        self._incoming: queue.Queue[ServingRequest] = queue.Queue()
        self._active: list[ServingRequest] = []
        self._stop = threading.Event()
        # Set by stop() to mean "finish whatever's already _active, but admit
        # nothing new" -- distinct from _stop, which halts _run()'s loop
        # entirely. See stop()'s own docstring for the full drain sequence.
        self._draining = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        # Multi-turn session registry. Mutated both from HTTP-handler threads
        # (create_session/delete_session/submit_turn) and from this
        # scheduler's own background thread (_retire, called from inside
        # _run/_run_once) -- guarded by one lock.
        self._sessions: dict[str, SessionState] = {}
        self._busy_sessions: set[str] = set()
        self._sessions_lock = threading.Lock()

        # Prometheus metrics. Every metric carries a "device" label (constant
        # per instance, = device_label) in addition to whatever label it
        # already has -- this is what lets several per-GPU schedulers share
        # ONE registry (see metrics_registry below) and still be told apart on
        # scrape, instead of colliding on the same metric name. On its own
        # CollectorRegistry by default rather than the library's process-global
        # one -- a process can (and this project's own benches do) construct
        # several schedulers at once, and a shared global registry would mix
        # their counts together unless a registry is explicitly passed in
        # (which pager_hf.serving.multi_gpu.MultiGpuScheduler's caller does,
        # precisely so its per-GPU schedulers land in one shared registry).
        self.metrics_registry = metrics_registry or CollectorRegistry()
        self._requests_total = Counter(
            "pager_hf_requests_total",
            "Completions requests retired, by outcome",
            ["outcome", "device"],
            registry=self.metrics_registry,
        )
        self._tokens_generated_total = Counter(
            "pager_hf_tokens_generated_total",
            "Tokens generated across all requests",
            ["device"],
            registry=self.metrics_registry,
        ).labels(device=device_label)
        self._decode_round_seconds = Histogram(
            "pager_hf_decode_round_seconds",
            "Wall time of one _step_single/_step_batched call",
            ["device"],
            registry=self.metrics_registry,
        ).labels(device=device_label)
        self._step_calls_total = Counter(
            "pager_hf_step_calls_total",
            "Decode-step calls, by mode",
            ["mode", "device"],
            registry=self.metrics_registry,
        )
        Gauge(
            "pager_hf_active_requests",
            "Requests currently admitted into the round-robin set",
            ["device"],
            registry=self.metrics_registry,
        ).labels(device=device_label).set_function(lambda: len(self._active))
        Gauge(
            "pager_hf_queued_requests",
            "Requests submitted but not yet admitted",
            ["device"],
            registry=self.metrics_registry,
        ).labels(device=device_label).set_function(self._incoming.qsize)
        Gauge(
            "pager_hf_active_sessions",
            "Persistent multi-turn sessions currently registered",
            ["device"],
            registry=self.metrics_registry,
        ).labels(device=device_label).set_function(lambda: len(self._sessions))

    def start(self) -> None:
        self._thread.start()
        logger.info(
            "ContinuousBatchingScheduler started: max_concurrent_sessions=%d max_queue_depth=%d "
            "vram_budget_per_session=%d ram_budget_per_session=%d batched_decode=%s",
            self._max_concurrent_sessions,
            self._max_queue_depth,
            self._vram_budget_per_session,
            self._ram_budget_per_session,
            self._can_batch,
        )

    def health(self) -> dict:
        """Snapshot for a health-check endpoint. thread_alive=False means the background
        loop died (an unhandled exception escaped _run()) -- the one silent-failure mode
        where the HTTP layer would otherwise look fine while nothing is actually being
        processed; a caller should treat that as unhealthy (HTTP 503), not just log it."""
        return {
            "status": "ok" if self._thread.is_alive() else "unhealthy",
            "thread_alive": self._thread.is_alive(),
            "active_requests": len(self._active),
            "queued_requests": self._incoming.qsize(),
            "active_sessions": len(self._sessions),
        }

    def stop(self, timeout: float | None = None) -> None:
        """
        Graceful shutdown: stop admitting new requests, but let anything already in
        _active finish naturally first (up to `timeout` seconds; None waits indefinitely,
        matching Thread.join()'s own convention). If the timeout elapses first, the
        background thread is force-stopped after its current round, and anything still
        stuck in _active or _incoming is explicitly failed with a clear error instead of
        being left to hang forever -- every caller gets a bounded, defined outcome.

        Backward compatible with every existing caller: this project's own bench scripts
        only ever call stop() after confirming (via req.done.wait()) that every request
        they submitted has already finished, so _active is already empty by then and this
        drain is an instant no-op for them.
        """
        self._draining.set()
        self._thread.join(timeout)

        if self._thread.is_alive():
            self._stop.set()
            self._thread.join()

        for req in list(self._active):
            req.error = RuntimeError("scheduler shut down before this request finished (drain timeout exceeded)")
            self._retire(req)

        while True:
            try:
                req = self._incoming.get_nowait()
            except queue.Empty:
                break
            req.error = RuntimeError("scheduler shut down before this request could be admitted")
            req.done.set()

    def load(self) -> int:
        """Accepted-but-not-yet-finished work: active + still queued. The least-loaded
        signal pager_hf.serving.multi_gpu.MultiGpuScheduler routes on; also what
        _check_capacity bounds. _incoming.qsize() is an approximate count under
        concurrent access (queue.Queue's own documented caveat) -- fine for a soft
        load/backpressure signal, not meant as a hard consistency guarantee."""
        return len(self._active) + self._incoming.qsize()

    def _check_capacity(self) -> None:
        """Backpressure: reject with SchedulerAtCapacityError once load() reaches
        max_concurrent_sessions + max_queue_depth, instead of letting the incoming
        queue grow without bound."""
        depth = self.load()
        limit = self._max_concurrent_sessions + self._max_queue_depth
        if depth >= limit:
            raise SchedulerAtCapacityError(f"scheduler is at capacity ({depth}/{limit} requests active or queued)")

    def _new_paged_model(self) -> PagedModel:
        return PagedModel(
            self._model,
            vram_budget=self._vram_budget_per_session,
            ram_budget=self._ram_budget_per_session,
            **self._paged_model_kwargs,
        )

    def _enqueue(
        self,
        paged_model: PagedModel,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        session_id: str | None,
    ) -> ServingRequest:
        req = ServingRequest(
            request_id=str(uuid.uuid4()),
            paged_model=paged_model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            session_id=session_id,
        )
        self._incoming.put(req)
        return req

    def submit(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> ServingRequest:
        """Enqueue a new, one-shot request (its own fresh PagedModel); admitted into the active
        round-robin set as soon as a slot is free. For a multi-turn conversation that keeps its
        PagedModel (and KV cache) alive across calls, use create_session()/submit_turn() instead.
        Raises SchedulerAtCapacityError if the queue is already at max_queue_depth."""
        self._check_capacity()
        return self._enqueue(
            self._new_paged_model(),
            input_ids.to(self._device),
            attention_mask.to(self._device),
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            session_id=None,
        )

    def create_session(self) -> str:
        """Allocate a new persistent PagedModel session; returns its session_id.
        Raises SessionCapacityError if the max_concurrent_sessions-sized pool is already full."""
        with self._sessions_lock:
            if len(self._sessions) >= self._max_concurrent_sessions:
                raise SessionCapacityError(
                    f"session registry is full ({len(self._sessions)}/{self._max_concurrent_sessions} sessions)"
                )
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = SessionState(session_id=session_id, paged_model=self._new_paged_model())
            return session_id

    def delete_session(self, session_id: str) -> None:
        """Drop a persistent session, freeing its PagedModel/KV cache once nothing else references it.
        Raises KeyError if unknown, SessionBusyError if it has an in-flight turn right now."""
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise KeyError(session_id)
            if session_id in self._busy_sessions:
                raise SessionBusyError(f"session {session_id!r} has an in-flight turn; wait for it to finish first")
            del self._sessions[session_id]

    def submit_turn(
        self,
        session_id: str,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> ServingRequest:
        """
        Enqueue the next turn of an existing multi-turn session, reusing its PagedModel instance
        instead of constructing a new one. input_ids/attention_mask are this turn's own new tokens
        (e.g. the user's next message) -- combined here with the session's growing history (None on
        the session's first turn) into the full sequence PagedModel._extend_session expects.

        Raises KeyError if session_id is unknown, SessionBusyError if the session already has an
        in-flight turn (one turn at a time per session, same as PagedModel.generate()'s own
        single-flight lock -- this just surfaces the conflict earlier and more clearly), or
        SchedulerAtCapacityError if the queue is already at max_queue_depth (checked first, before
        any other side effect, so a rejected turn never leaves the session incorrectly marked busy).
        """
        self._check_capacity()
        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if session_id in self._busy_sessions:
                raise SessionBusyError(f"session {session_id!r} already has an in-flight turn")
            self._busy_sessions.add(session_id)

            if session.input_ids is None:
                combined_input_ids, combined_attention_mask = input_ids, attention_mask
            else:
                combined_input_ids = torch.cat([session.input_ids, input_ids], dim=1)
                combined_attention_mask = torch.cat([session.attention_mask, attention_mask], dim=1)

        return self._enqueue(
            session.paged_model,
            combined_input_ids,
            combined_attention_mask,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            session_id=session_id,
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._draining.is_set():
                self._admit_new_requests()

            if not self._active:
                if self._draining.is_set():
                    return  # drained: nothing active, nothing more will be admitted
                self._stop.wait(timeout=0.005)
                continue

            self._run_once()

    def _run_once(self) -> None:
        """One scheduling round over the current _active snapshot: fresh sessions
        (no store yet) and any already-started session without a tail-length
        batching partner this round fall back to _step_single; everyone else is
        grouped by tail length and advanced via _step_batched. Split out from
        _run()'s infinite loop so a single round is directly testable."""
        snapshot = list(self._active)  # _active is mutated (removals) during this pass

        if not self._can_batch:
            for req in snapshot:
                self._step_single(req)
            return

        fresh = [req for req in snapshot if req.paged_model._store is None]
        started = [req for req in snapshot if req.paged_model._store is not None]

        for req in fresh:
            self._step_single(req)

        groups: dict[int, list[ServingRequest]] = {}
        for req in started:
            tail_len = req.paged_model._tail_past[0][0].shape[2]
            groups.setdefault(tail_len, []).append(req)

        for group in groups.values():
            if len(group) >= 2:
                self._step_batched(group)
            else:
                self._step_single(group[0])

    def _admit_new_requests(self) -> None:
        while len(self._active) < self._max_concurrent_sessions:
            try:
                req = self._incoming.get_nowait()
            except queue.Empty:
                return
            self._active.append(req)

    def _step_single(self, req: ServingRequest) -> None:
        """One request's own generate() call: always used for a fresh session's first
        step (prefill + first decode token in one call), and as the fallback for any
        already-started session that doesn't share a tail-length batching group with
        another active session this round."""
        self._step_calls_total.labels(mode="single", device=self._device_label).inc()
        started = time.monotonic()
        try:
            new_tokens = req.paged_model.generate(
                input_ids=req.input_ids,
                attention_mask=req.attention_mask,
                max_new_tokens=1,
                do_sample=req.do_sample,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced to the HTTP layer via req.error, not swallowed
            logger.error("request %s failed during decode step: %s", req.request_id, exc)
            req.error = exc
            self._retire(req)
            return
        finally:
            self._decode_round_seconds.observe(time.monotonic() - started)

        self._finish_step(req, new_tokens[-1])

    def _step_batched(self, group: list[ServingRequest]) -> None:
        """Stage 2: advance every request in `group` (already-started, same tail
        length) with a single combined model.forward() call instead of one per
        request."""
        self._step_calls_total.labels(mode="batched", device=self._device_label).inc()
        started = time.monotonic()
        sessions = [req.paged_model for req in group]
        next_tokens = [int(req.input_ids[0, -1].item()) for req in group]

        try:
            logits_list = batched_decode_step(sessions, next_tokens, self._device)
        except Exception as exc:  # noqa: BLE001 -- same as _step_single: surfaced via req.error, not swallowed
            logger.error("batched decode step failed for %d requests: %s", len(group), exc)
            for req in group:
                req.error = exc
                self._retire(req)
            return
        finally:
            self._decode_round_seconds.observe(time.monotonic() - started)

        for req, logits in zip(group, logits_list):
            if req.do_sample:
                token_id = int(
                    PagedModel._sample_next_token(
                        logits.unsqueeze(0),
                        temperature=req.temperature,
                        top_k=req.top_k,
                        top_p=req.top_p,
                        generator=None,
                    ).item()
                )
            else:
                token_id = int(torch.argmax(logits).item())
            self._finish_step(req, token_id)

    def _finish_step(self, req: ServingRequest, token_id: int) -> None:
        """Shared bookkeeping after either stepping path produces this round's token:
        record it, grow the request's own input_ids/attention_mask (kept accurate
        even for batched-path requests, so a session can fall back to _step_single
        cleanly in a later round if it ends up alone in its tail-length group), and
        retire on EOS or max_new_tokens."""
        req.generated_tokens.append(token_id)
        req.output_queue.put(token_id)
        self._tokens_generated_total.inc()

        req.input_ids = torch.cat(
            [req.input_ids, torch.tensor([[token_id]], dtype=req.input_ids.dtype, device=req.input_ids.device)], dim=1
        )
        req.attention_mask = torch.cat(
            [req.attention_mask, torch.ones((1, 1), dtype=req.attention_mask.dtype, device=req.attention_mask.device)],
            dim=1,
        )

        is_eos = req.eos_token_id is not None and token_id == req.eos_token_id
        if is_eos or len(req.generated_tokens) >= req.max_new_tokens:
            self._retire(req)

    def _retire(self, req: ServingRequest) -> None:
        req.output_queue.put(None)  # sentinel: stream finished
        req.done.set()
        self._requests_total.labels(
            outcome="error" if req.error is not None else "success", device=self._device_label
        ).inc()
        if req in self._active:
            self._active.remove(req)

        if req.session_id is not None:
            with self._sessions_lock:
                self._busy_sessions.discard(req.session_id)
                session = self._sessions.get(req.session_id)
                if session is not None:
                    # req.input_ids/attention_mask already reflect the full new
                    # conversation state -- _finish_step grew them by one token
                    # per decode step above -- so the next turn's submit_turn()
                    # call extends from exactly here.
                    session.input_ids = req.input_ids
                    session.attention_mask = req.attention_mask
