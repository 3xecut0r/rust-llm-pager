from __future__ import annotations

from prometheus_client import CollectorRegistry

from pager_hf.serving.multi_gpu import MultiGpuScheduler
from pager_hf.serving.scheduler import SessionCapacityError

# CPU-only, fake-per-GPU-scheduler tests for MultiGpuScheduler's own routing
# logic (least-loaded submit, session affinity, capacity fallback, health/metrics
# aggregation) -- deliberately not exercising real ContinuousBatchingScheduler
# instances or real devices here. Real multi-GPU hardware verification (two
# real GPUs, two real model replicas, real routed traffic) needs a rented
# multi-GPU instance and is not attempted in this pass -- see the README/plan
# notes on this being an explicitly disclosed, still-pending verification.


class _FakeGpuScheduler:
    def __init__(self, name: str, *, load: int = 0, session_capacity: int | None = None):
        self.name = name
        self._load = load
        self._session_capacity = session_capacity
        self.submitted: list[tuple] = []
        self.turns_submitted: list[tuple] = []
        self.deleted_sessions: list[str] = []
        self._sessions: set[str] = set()
        self._next_session_index = 0
        self._thread_alive = True
        self.metrics_registry = CollectorRegistry()

    def load(self) -> int:
        return self._load

    def health(self) -> dict:
        return {
            "status": "ok" if self._thread_alive else "unhealthy",
            "thread_alive": self._thread_alive,
            "active_requests": self._load,
            "queued_requests": 0,
            "active_sessions": len(self._sessions),
        }

    def submit(self, *args, **kwargs):
        self.submitted.append((args, kwargs))
        return f"{self.name}:request"

    def create_session(self) -> str:
        if self._session_capacity is not None and len(self._sessions) >= self._session_capacity:
            raise SessionCapacityError(f"{self.name} session registry full")
        session_id = f"{self.name}:session:{self._next_session_index}"
        self._next_session_index += 1
        self._sessions.add(session_id)
        return session_id

    def submit_turn(self, session_id, *args, **kwargs):
        assert session_id in self._sessions
        self.turns_submitted.append((session_id, args, kwargs))
        return f"{self.name}:turn:{session_id}"

    def delete_session(self, session_id) -> None:
        self._sessions.discard(session_id)
        self.deleted_sessions.append(session_id)


def test_submit_routes_to_least_loaded_scheduler():
    busy = _FakeGpuScheduler("busy", load=5)
    idle = _FakeGpuScheduler("idle", load=0)
    router = MultiGpuScheduler([busy, idle])

    result = router.submit("ids", "mask", max_new_tokens=1)

    assert result == "idle:request"
    assert idle.submitted and not busy.submitted


def test_create_session_falls_back_when_least_loaded_gpu_is_full():
    """The least-loaded GPU by decode load might still have its OWN session
    registry full (an independent per-GPU cap) -- the router must try the
    next-least-loaded one instead of failing outright."""
    idle_but_full = _FakeGpuScheduler("idle_but_full", load=0, session_capacity=0)
    busier_but_has_room = _FakeGpuScheduler("busier_but_has_room", load=5, session_capacity=10)
    router = MultiGpuScheduler([idle_but_full, busier_but_has_room])

    session_id = router.create_session()

    assert session_id.startswith("busier_but_has_room:")
    assert session_id in busier_but_has_room._sessions


def test_create_session_raises_when_every_gpu_is_full():
    a = _FakeGpuScheduler("a", session_capacity=0)
    b = _FakeGpuScheduler("b", session_capacity=0)
    router = MultiGpuScheduler([a, b])

    try:
        router.create_session()
        assert False, "expected SessionCapacityError"
    except SessionCapacityError:
        pass


def test_submit_turn_and_delete_session_route_to_the_owning_gpu_not_the_least_loaded_one():
    gpu_a = _FakeGpuScheduler("a", load=0)
    gpu_b = _FakeGpuScheduler("b", load=0)
    router = MultiGpuScheduler([gpu_a, gpu_b])

    session_id = router.create_session()  # lands on whichever is "least loaded" first (gpu_a, tie broken by order)
    owner = gpu_a if session_id.startswith("a:") else gpu_b
    other = gpu_b if owner is gpu_a else gpu_a

    # Make the OTHER gpu look far less loaded now -- the turn must still go to the owner.
    other._load = 0
    owner._load = 100

    result = router.submit_turn(session_id, "ids", "mask", max_new_tokens=1)
    assert result == f"{owner.name}:turn:{session_id}"
    assert owner.turns_submitted and not other.turns_submitted

    router.delete_session(session_id)
    assert owner.deleted_sessions == [session_id]
    assert other.deleted_sessions == []


def test_submit_turn_and_delete_session_raise_key_error_for_unknown_session():
    router = MultiGpuScheduler([_FakeGpuScheduler("a")])
    try:
        router.submit_turn("ghost", "ids", "mask", max_new_tokens=1)
        assert False, "expected KeyError"
    except KeyError:
        pass

    try:
        router.delete_session("ghost")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_health_aggregates_counts_and_is_unhealthy_if_any_gpu_is_unhealthy():
    gpu_a = _FakeGpuScheduler("a", load=2)
    gpu_b = _FakeGpuScheduler("b", load=3)
    router = MultiGpuScheduler([gpu_a, gpu_b])
    router.create_session()

    status = router.health()
    assert status["status"] == "ok"
    assert status["active_requests"] == 5  # 2 + 3
    assert status["active_sessions"] == 1
    assert len(status["gpus"]) == 2

    gpu_b._thread_alive = False
    status = router.health()
    assert status["status"] == "unhealthy"


def test_requires_at_least_one_scheduler():
    try:
        MultiGpuScheduler([])
        assert False, "expected ValueError"
    except ValueError:
        pass
