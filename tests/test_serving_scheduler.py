from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
import torch

import pager_hf.serving.scheduler as scheduler_module
from pager_hf.serving.scheduler import (
    ContinuousBatchingScheduler,
    SchedulerAtCapacityError,
    SessionBusyError,
    SessionCapacityError,
)

# Deliberately does not start the scheduler's background thread anywhere in
# this file -- _admit_new_requests()/_step_single() are driven directly, one
# call at a time, so round-robin ordering is fully deterministic instead of
# racing real wall-clock timing. These tests exercise _step_single (the
# per-request fallback path) directly, not _run()'s tail-length grouping /
# _step_batched (Stage 2) -- the fake PagedModel here has no _store/_tail_past
# to group by; batched_decode_step's own behavior is covered separately
# (tests/test_batched_decode.py, bench/real_kv_batched_decode_mvp.py).

EOS_TOKEN_ID = 999


class _FakePagedModel:
    def __init__(self, token_sequence: list[int]):
        self._tokens = iter(token_sequence)
        self._store = None  # "fresh": _run_once() must always route this through _step_single

    def generate(self, *, input_ids, attention_mask, max_new_tokens, **kwargs):
        assert max_new_tokens == 1
        return [next(self._tokens)]


class _FakePagedModelFactory:
    """Drop-in replacement for the PagedModel class: each call constructs one
    fake session that yields the next prescribed token sequence, in the order
    scheduler.submit() constructs them."""

    def __init__(self, token_sequences: list[list[int]]):
        self._token_sequences = list(token_sequences)
        self._next_index = 0

    def __call__(self, model, *, vram_budget, ram_budget, **kwargs):
        seq = self._token_sequences[self._next_index]
        self._next_index += 1
        return _FakePagedModel(seq)


def make_scheduler(monkeypatch, token_sequences: list[list[int]], **overrides) -> ContinuousBatchingScheduler:
    monkeypatch.setattr(scheduler_module, "PagedModel", _FakePagedModelFactory(token_sequences))
    kwargs = dict(
        model=object(),
        tokenizer=SimpleNamespace(eos_token_id=EOS_TOKEN_ID),
        max_concurrent_sessions=2,
        total_vram_budget=2_000_000,
        total_ram_budget=2_000_000,
    )
    kwargs.update(overrides)
    return ContinuousBatchingScheduler(**kwargs)


def make_ids() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long)


def test_rejects_invalid_max_concurrent_sessions():
    with pytest.raises(ValueError, match="max_concurrent_sessions"):
        ContinuousBatchingScheduler(
            model=object(),
            tokenizer=SimpleNamespace(eos_token_id=None),
            max_concurrent_sessions=0,
            total_vram_budget=1000,
            total_ram_budget=1000,
        )


def test_admission_respects_max_concurrent_sessions(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, 2], [3, 4], [5, 6]])
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=2)
    req_b = scheduler.submit(ids, mask, max_new_tokens=2)
    req_c = scheduler.submit(ids, mask, max_new_tokens=2)

    scheduler._admit_new_requests()
    assert scheduler._active == [req_a, req_b]  # only 2 slots -- req_c stays queued, not rejected

    scheduler._step_single(req_a)
    scheduler._step_single(req_a)
    assert req_a.done.is_set()
    assert req_a not in scheduler._active

    scheduler._admit_new_requests()
    assert scheduler._active == [req_b, req_c]  # a slot freed up, req_c admitted


def test_round_robin_interleaves_not_sequential(monkeypatch):
    """Both requests must make progress after one round each -- neither runs to
    completion before the other gets a turn."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, 2, 3], [4, 5, 6]])
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=3)
    req_b = scheduler.submit(ids, mask, max_new_tokens=3)
    scheduler._admit_new_requests()

    scheduler._step_single(req_a)
    scheduler._step_single(req_b)

    assert req_a.generated_tokens == [1]
    assert req_b.generated_tokens == [4]
    assert not req_a.done.is_set()
    assert not req_b.done.is_set()


def test_eos_retires_request_before_max_new_tokens(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, EOS_TOKEN_ID, 3]])
    ids, mask = make_ids()
    req = scheduler.submit(ids, mask, max_new_tokens=10)
    scheduler._admit_new_requests()

    scheduler._step_single(req)
    assert not req.done.is_set()

    scheduler._step_single(req)
    assert req.done.is_set()
    assert req.generated_tokens == [1, EOS_TOKEN_ID]
    assert req not in scheduler._active


def test_output_queue_drains_to_sentinel(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[7, 8]])
    ids, mask = make_ids()
    req = scheduler.submit(ids, mask, max_new_tokens=2)
    scheduler._admit_new_requests()

    scheduler._step_single(req)
    scheduler._step_single(req)

    assert req.output_queue.get_nowait() == 7
    assert req.output_queue.get_nowait() == 8
    assert req.output_queue.get_nowait() is None


class _FakeStartedPagedModel(_FakePagedModel):
    """A _FakePagedModel that already has an active session (_store is not
    None) at a given tail length -- enough state for _run_once()'s grouping
    logic to route it toward _step_batched, without needing a real model."""

    def __init__(self, token_sequence: list[int], tail_len: int):
        super().__init__(token_sequence)
        self.model = object()
        self._store = object()
        self._tail_past = [(torch.zeros(1, 1, tail_len, 1), torch.zeros(1, 1, tail_len, 1))]


def test_run_once_routes_fresh_sessions_to_step_single(monkeypatch):
    """A session with no active store yet (_store is None, like _FakePagedModel's
    default) must always go through _step_single, even when batching is enabled --
    prefill + first decode token always happens via one generate() call."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2]])
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=1)
    req_b = scheduler.submit(ids, mask, max_new_tokens=1)
    scheduler._admit_new_requests()

    step_single_calls = []
    monkeypatch.setattr(scheduler, "_step_single", lambda req: step_single_calls.append(req))
    monkeypatch.setattr(scheduler, "_step_batched", lambda group: pytest.fail("should not batch fresh sessions"))

    scheduler._run_once()

    assert {id(r) for r in step_single_calls} == {id(req_a), id(req_b)}


def test_run_once_batches_started_sessions_sharing_a_tail_length(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2], [3]])
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=1)
    req_b = scheduler.submit(ids, mask, max_new_tokens=1)
    scheduler.submit(ids, mask, max_new_tokens=1)  # stays queued (max_concurrent_sessions=2)
    scheduler._admit_new_requests()

    req_a.paged_model = _FakeStartedPagedModel([1], tail_len=5)
    req_b.paged_model = _FakeStartedPagedModel([2], tail_len=5)

    batched_groups = []
    monkeypatch.setattr(scheduler, "_step_batched", lambda group: batched_groups.append(group))
    monkeypatch.setattr(scheduler, "_step_single", lambda req: pytest.fail("should not single-step a matched pair"))

    scheduler._run_once()

    assert len(batched_groups) == 1
    assert {id(r) for r in batched_groups[0]} == {id(req_a), id(req_b)}


def test_run_once_falls_back_to_single_step_for_a_lone_tail_length(monkeypatch):
    """A started session with no batching partner this round (its tail length
    doesn't match any other active session's) must still make progress via
    _step_single, not get stuck waiting for a partner that isn't there."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2]])
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=1)
    req_b = scheduler.submit(ids, mask, max_new_tokens=1)
    scheduler._admit_new_requests()

    req_a.paged_model = _FakeStartedPagedModel([1], tail_len=3)
    req_b.paged_model = _FakeStartedPagedModel([2], tail_len=9)  # different tail length -- no partner

    step_single_calls = []
    monkeypatch.setattr(scheduler, "_step_single", lambda req: step_single_calls.append(req))
    monkeypatch.setattr(scheduler, "_step_batched", lambda group: pytest.fail("should not batch a lone session"))

    scheduler._run_once()

    assert {id(r) for r in step_single_calls} == {id(req_a), id(req_b)}


def test_create_session_respects_capacity_and_returns_unique_ids(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[], []], max_concurrent_sessions=2)

    id_a = scheduler.create_session()
    id_b = scheduler.create_session()
    assert id_a != id_b
    assert set(scheduler._sessions) == {id_a, id_b}

    with pytest.raises(SessionCapacityError):
        scheduler.create_session()


def test_submit_turn_unknown_session_raises_key_error(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[])
    ids, mask = make_ids()
    with pytest.raises(KeyError):
        scheduler.submit_turn("no-such-session", ids, mask, max_new_tokens=1)


def test_submit_turn_reuses_same_paged_model_across_turns(monkeypatch):
    """Unlike submit() (a fresh PagedModel every call), turns for one session must all
    share the identical PagedModel instance -- that's the whole point of a session."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, 2]])
    session_id = scheduler.create_session()
    ids, mask = make_ids()

    req_1 = scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)
    scheduler._admit_new_requests()
    scheduler._step_single(req_1)
    assert req_1.done.is_set()

    req_2 = scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)
    assert req_2.paged_model is req_1.paged_model


def test_submit_turn_combines_previous_turn_output_with_new_turn_input(monkeypatch):
    """The second turn's constructed input_ids must be [first turn's full final
    input_ids (original + generated tokens)] ++ [second turn's own new tokens] --
    exactly what PagedModel._extend_session expects as the growing sequence."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[42]])
    session_id = scheduler.create_session()
    first_ids, first_mask = make_ids()

    req_1 = scheduler.submit_turn(session_id, first_ids, first_mask, max_new_tokens=1)
    scheduler._admit_new_requests()
    scheduler._step_single(req_1)
    assert req_1.done.is_set()
    expected_prefix = req_1.input_ids.clone()  # first_ids ++ generated token [42]

    second_ids = torch.tensor([[7, 8]], dtype=torch.long)
    second_mask = torch.ones_like(second_ids)
    req_2 = scheduler.submit_turn(session_id, second_ids, second_mask, max_new_tokens=1)

    assert torch.equal(req_2.input_ids, torch.cat([expected_prefix, second_ids], dim=1))
    assert torch.equal(req_2.attention_mask, torch.cat([req_1.attention_mask, second_mask], dim=1))


def test_submit_turn_rejects_concurrent_turn_on_same_session(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, 2]])
    session_id = scheduler.create_session()
    ids, mask = make_ids()

    scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)  # left in-flight, never stepped/retired
    with pytest.raises(SessionBusyError):
        scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)


def test_delete_session_rejects_in_flight_session_then_succeeds_after(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[9]])
    session_id = scheduler.create_session()
    ids, mask = make_ids()

    req = scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)
    with pytest.raises(SessionBusyError):
        scheduler.delete_session(session_id)

    scheduler._admit_new_requests()
    scheduler._step_single(req)
    assert req.done.is_set()

    scheduler.delete_session(session_id)
    assert session_id not in scheduler._sessions

    with pytest.raises(KeyError):
        scheduler.delete_session(session_id)


def test_run_once_never_batches_when_streaming_attention_is_disabled(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2]], use_streaming_attention=False)
    ids, mask = make_ids()
    req_a = scheduler.submit(ids, mask, max_new_tokens=1)
    req_b = scheduler.submit(ids, mask, max_new_tokens=1)
    scheduler._admit_new_requests()

    req_a.paged_model = _FakeStartedPagedModel([1], tail_len=5)
    req_b.paged_model = _FakeStartedPagedModel([2], tail_len=5)  # would match if batching were enabled

    step_single_calls = []
    monkeypatch.setattr(scheduler, "_step_single", lambda req: step_single_calls.append(req))
    monkeypatch.setattr(scheduler, "_step_batched", lambda group: pytest.fail("batching must be off entirely"))

    scheduler._run_once()

    assert {id(r) for r in step_single_calls} == {id(req_a), id(req_b)}


class _FakeNeverEosPagedModel:
    """A fake session whose generate() always makes progress but never hits EOS or
    naturally runs out -- used to exercise stop(timeout=...)'s force-retire path
    without genuinely hanging the test (each call returns immediately)."""

    def __init__(self):
        self._store = None
        self._counter = itertools.count(start=10_000)  # well above EOS_TOKEN_ID, so it's never accidentally hit

    def generate(self, *, input_ids, attention_mask, max_new_tokens, **kwargs):
        assert max_new_tokens == 1
        return [next(self._counter)]


def test_health_reports_thread_status():
    scheduler = ContinuousBatchingScheduler(
        model=object(),
        tokenizer=SimpleNamespace(eos_token_id=EOS_TOKEN_ID),
        max_concurrent_sessions=2,
        total_vram_budget=2_000_000,
        total_ram_budget=2_000_000,
    )
    assert scheduler.health()["thread_alive"] is False  # never started

    scheduler.start()
    assert scheduler.health()["thread_alive"] is True

    scheduler.stop()
    assert scheduler.health() == {
        "status": "unhealthy",
        "thread_alive": False,
        "active_requests": 0,
        "queued_requests": 0,
        "active_sessions": 0,
    }


def test_stop_drains_active_request_to_completion(monkeypatch):
    """stop() with no timeout (the default) must wait for an already-admitted request
    to finish naturally, not abandon it mid-generation -- the actual point of a
    'graceful' shutdown, unlike the old stop() this replaces."""
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1, 2, 3]])
    ids, mask = make_ids()
    req = scheduler.submit(ids, mask, max_new_tokens=3)

    scheduler.start()
    scheduler.stop()

    assert req.done.is_set()
    assert req.error is None
    assert req.generated_tokens == [1, 2, 3]


def test_stop_with_timeout_force_retires_a_request_that_never_finishes(monkeypatch):
    """A request that keeps making progress but never reaches EOS/max_new_tokens
    within the given timeout must still be given a definite, clean outcome --
    retired with an explicit error, not left to hang forever."""
    monkeypatch.setattr(
        scheduler_module, "PagedModel", lambda model, *, vram_budget, ram_budget, **kwargs: _FakeNeverEosPagedModel()
    )
    scheduler = ContinuousBatchingScheduler(
        model=object(),
        tokenizer=SimpleNamespace(eos_token_id=EOS_TOKEN_ID),
        max_concurrent_sessions=2,
        total_vram_budget=2_000_000,
        total_ram_budget=2_000_000,
    )
    ids, mask = make_ids()
    req = scheduler.submit(ids, mask, max_new_tokens=1_000_000)  # will not finish naturally

    scheduler.start()
    scheduler.stop(timeout=0.05)

    assert not scheduler._thread.is_alive()
    assert req.done.is_set()
    assert req.error is not None


def test_submit_raises_at_capacity(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2]], max_concurrent_sessions=1, max_queue_depth=0)
    ids, mask = make_ids()
    scheduler.submit(ids, mask, max_new_tokens=1)  # fills the only slot (queued, thread never started)

    with pytest.raises(SchedulerAtCapacityError):
        scheduler.submit(ids, mask, max_new_tokens=1)


def test_submit_turn_at_capacity_does_not_mark_session_busy(monkeypatch):
    scheduler = make_scheduler(monkeypatch, token_sequences=[[1], [2]], max_concurrent_sessions=1, max_queue_depth=0)
    session_id = scheduler.create_session()
    ids, mask = make_ids()
    scheduler.submit(ids, mask, max_new_tokens=1)  # fills capacity via the one-shot path

    with pytest.raises(SchedulerAtCapacityError):
        scheduler.submit_turn(session_id, ids, mask, max_new_tokens=1)

    assert session_id not in scheduler._busy_sessions
