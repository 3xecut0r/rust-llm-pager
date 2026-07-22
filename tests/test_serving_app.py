from __future__ import annotations

import queue
import threading

from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter

from pager_hf.serving.app import create_app
from pager_hf.serving.scheduler import SchedulerAtCapacityError, SessionBusyError, SessionCapacityError

# CPU-only, fake-scheduler tests for the HTTP layer itself (auth gating, error-to-status-code
# mapping, health/metrics shape) -- deliberately not exercising a real ContinuousBatchingScheduler
# or model here, since none of that is what this layer's own logic depends on. Real end-to-end
# behavior (a real model, a real scheduler, real generation) is covered separately in
# bench/serving_concurrent_requests_mvp.py.


class _FakeEncoded(dict):
    pass


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt"):
        return _FakeEncoded(input_ids=f"ids:{text}", attention_mask=f"mask:{text}")

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded:{token_ids}"


class _FakeRequest:
    def __init__(self, generated_tokens=None, session_id=None, error=None):
        self.request_id = "req-1"
        self.session_id = session_id
        self.generated_tokens = generated_tokens or []
        self.error = error
        self.done = threading.Event()
        self.done.set()
        self.output_queue: "queue.Queue[int | None]" = queue.Queue()
        self.output_queue.put(None)


class _FakeScheduler:
    def __init__(self, *, submit_result=None, submit_exception=None, health_status=None, metrics_registry=None):
        self.submit_result = submit_result
        self.submit_exception = submit_exception
        self._health_status = health_status or {
            "status": "ok",
            "thread_alive": True,
            "active_requests": 0,
            "queued_requests": 0,
            "active_sessions": 0,
        }
        self.metrics_registry = metrics_registry or CollectorRegistry()

    def health(self):
        return self._health_status

    def _submit_like(self):
        if self.submit_exception is not None:
            raise self.submit_exception
        return self.submit_result

    def submit(self, input_ids, attention_mask, max_new_tokens, **kwargs):
        return self._submit_like()

    def submit_turn(self, session_id, input_ids, attention_mask, max_new_tokens, **kwargs):
        return self._submit_like()

    def create_session(self):
        if self.submit_exception is not None:
            raise self.submit_exception
        return "session-123"

    def delete_session(self, session_id):
        if self.submit_exception is not None:
            raise self.submit_exception


def make_client(scheduler, api_key=None) -> TestClient:
    app = create_app(scheduler, _FakeTokenizer(), api_key=api_key)
    return TestClient(app)


def test_healthz_ok():
    client = make_client(_FakeScheduler())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_healthz_unhealthy_returns_503():
    scheduler = _FakeScheduler(
        health_status={
            "status": "unhealthy",
            "thread_alive": False,
            "active_requests": 0,
            "queued_requests": 0,
            "active_sessions": 0,
        }
    )
    client = make_client(scheduler)
    assert client.get("/healthz").status_code == 503


def test_healthz_stays_open_even_when_api_key_is_configured():
    client = make_client(_FakeScheduler(), api_key="secret")
    assert client.get("/healthz").status_code == 200


def test_metrics_returns_prometheus_text_and_content_type():
    registry = CollectorRegistry()
    Counter("test_metric_total", "a probe metric", registry=registry).inc(3)
    client = make_client(_FakeScheduler(metrics_registry=registry))

    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "test_metric_total 3.0" in response.text


def test_completions_without_api_key_configured_succeeds():
    scheduler = _FakeScheduler(submit_result=_FakeRequest(generated_tokens=[1, 2, 3]))
    client = make_client(scheduler)

    response = client.post("/v1/completions", json={"prompt": "hello", "max_tokens": 3})
    assert response.status_code == 200
    assert response.json()["choices"][0]["text"] == "decoded:[1, 2, 3]"


def test_completions_requires_api_key_when_configured():
    scheduler = _FakeScheduler(submit_result=_FakeRequest(generated_tokens=[1]))
    client = make_client(scheduler, api_key="secret")

    assert client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1}).status_code == 401
    assert (
        client.post(
            "/v1/completions", json={"prompt": "hi", "max_tokens": 1}, headers={"Authorization": "Bearer wrong"}
        ).status_code
        == 401
    )
    ok = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 1}, headers={"Authorization": "Bearer secret"}
    )
    assert ok.status_code == 200


def test_metrics_requires_api_key_when_configured():
    client = make_client(_FakeScheduler(), api_key="secret")
    assert client.get("/metrics").status_code == 401
    assert client.get("/metrics", headers={"Authorization": "Bearer secret"}).status_code == 200


def test_completions_maps_unknown_session_to_404():
    scheduler = _FakeScheduler(submit_exception=KeyError("nope"))
    client = make_client(scheduler)
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1, "session_id": "ghost"})
    assert response.status_code == 404


def test_completions_maps_busy_session_to_409():
    scheduler = _FakeScheduler(submit_exception=SessionBusyError("already busy"))
    client = make_client(scheduler)
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1, "session_id": "s1"})
    assert response.status_code == 409


def test_completions_maps_capacity_error_to_429_with_retry_after():
    scheduler = _FakeScheduler(submit_exception=SchedulerAtCapacityError("full"))
    client = make_client(scheduler)
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 1})
    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"


def test_create_session_maps_capacity_error_to_503():
    scheduler = _FakeScheduler(submit_exception=SessionCapacityError("registry full"))
    client = make_client(scheduler)
    assert client.post("/v1/sessions").status_code == 503


def test_delete_session_maps_key_error_to_404_and_busy_error_to_409():
    client_404 = make_client(_FakeScheduler(submit_exception=KeyError("nope")))
    assert client_404.delete("/v1/sessions/ghost").status_code == 404

    client_409 = make_client(_FakeScheduler(submit_exception=SessionBusyError("busy")))
    assert client_409.delete("/v1/sessions/s1").status_code == 409
