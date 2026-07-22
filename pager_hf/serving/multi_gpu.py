from __future__ import annotations

import logging
import threading

from .scheduler import ContinuousBatchingScheduler, ServingRequest, SessionCapacityError

logger = logging.getLogger(__name__)


class MultiGpuScheduler:
    """
    Load-balances across N independent ContinuousBatchingSchedulers, each with
    its own full copy of the model on its own device -- data-parallel
    replicas, not model/tensor parallelism (a model that doesn't already fit
    on one GPU is out of scope, same standing constraint as everywhere else
    in this project). Duck-types the exact subset of ContinuousBatchingScheduler's
    public surface pager_hf.serving.app.create_app calls (submit, submit_turn,
    create_session, delete_session, health, metrics_registry, start, stop) so
    create_app works completely unmodified against either one scheduler or
    this router.

    A session stays on whichever GPU it was created on for its entire
    lifetime -- sessions are not portable across devices, and don't need to
    be; only fresh, one-shot submit() calls and new sessions get load-balanced.
    """

    def __init__(self, schedulers: list[ContinuousBatchingScheduler]):
        if not schedulers:
            raise ValueError("MultiGpuScheduler requires at least one underlying scheduler")
        self._schedulers = list(schedulers)
        self.metrics_registry = self._schedulers[0].metrics_registry

        self._session_owner: dict[str, ContinuousBatchingScheduler] = {}
        self._session_owner_lock = threading.Lock()

    def _by_load(self) -> list[ContinuousBatchingScheduler]:
        return sorted(self._schedulers, key=lambda s: s.load())

    def start(self) -> None:
        for scheduler in self._schedulers:
            scheduler.start()
        logger.info("MultiGpuScheduler started: %d GPU(s)", len(self._schedulers))

    def stop(self, timeout: float | None = None) -> None:
        # Sequential, not parallelized -- simple and correct; a slower shutdown
        # when several schedulers each have long-draining requests is a
        # disclosed inefficiency, not a correctness issue, for this pass.
        for scheduler in self._schedulers:
            scheduler.stop(timeout=timeout)

    def health(self) -> dict:
        per_gpu = [scheduler.health() for scheduler in self._schedulers]
        healthy = all(status["thread_alive"] for status in per_gpu)
        return {
            "status": "ok" if healthy else "unhealthy",
            "thread_alive": healthy,
            "active_requests": sum(status["active_requests"] for status in per_gpu),
            "queued_requests": sum(status["queued_requests"] for status in per_gpu),
            "active_sessions": sum(status["active_sessions"] for status in per_gpu),
            "gpus": per_gpu,
        }

    def submit(self, *args, **kwargs) -> ServingRequest:
        """Route a one-shot request to the least-loaded GPU."""
        return self._by_load()[0].submit(*args, **kwargs)

    def create_session(self) -> str:
        """
        Try the least-loaded GPU first; if its own session registry happens to
        already be full (a per-GPU cap independent of decode load), fall back to
        the next-least-loaded one instead of failing outright. Only raises
        SessionCapacityError if every GPU's registry is full.
        """
        last_error: SessionCapacityError | None = None
        for scheduler in self._by_load():
            try:
                session_id = scheduler.create_session()
            except SessionCapacityError as exc:
                last_error = exc
                continue
            with self._session_owner_lock:
                self._session_owner[session_id] = scheduler
            return session_id
        raise last_error

    def _owner(self, session_id: str) -> ContinuousBatchingScheduler:
        with self._session_owner_lock:
            owner = self._session_owner.get(session_id)
        if owner is None:
            raise KeyError(session_id)
        return owner

    def submit_turn(self, session_id: str, *args, **kwargs) -> ServingRequest:
        """Route to whichever GPU this session was created on -- never load-balanced,
        since a session's PagedModel/KV cache lives on one specific device."""
        return self._owner(session_id).submit_turn(session_id, *args, **kwargs)

    def delete_session(self, session_id: str) -> None:
        owner = self._owner(session_id)
        owner.delete_session(session_id)
        with self._session_owner_lock:
            self._session_owner.pop(session_id, None)
