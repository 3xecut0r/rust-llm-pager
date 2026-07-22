from __future__ import annotations

import asyncio
import json
import logging

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from .scheduler import ContinuousBatchingScheduler, SchedulerAtCapacityError, SessionBusyError, SessionCapacityError

logger = logging.getLogger(__name__)

_RETRY_AFTER_SECONDS = "1"


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    do_sample: bool = False
    stream: bool = False
    session_id: str | None = None


def create_app(scheduler: ContinuousBatchingScheduler, tokenizer, api_key: str | None = None) -> FastAPI:
    """
    Build the FastAPI app around an already-started ContinuousBatchingScheduler
    (or pager_hf.serving.multi_gpu.MultiGpuScheduler, which duck-types the same
    submit/submit_turn/create_session/delete_session/health/metrics_registry
    surface this app calls -- nothing here branches on which one it got).

    Tokenization stays on CPU; whichever scheduler ends up handling a request
    (the router may pick any of several, each on its own device) moves the
    tensors to its own device itself. This app never needs to know what device
    anything runs on.

    /v1/completions is loosely OpenAI-shaped (prompt in, text out) -- not a
    full OpenAI-compatible surface (no chat templating, no logprobs, no
    multiple choices per request). Passing no session_id gives a fresh,
    one-shot completion with its own throwaway PagedModel, same as always.
    Passing a session_id (from POST /v1/sessions) instead reuses that
    session's PagedModel and KV cache -- request.prompt is tokenized as just
    this turn's new text and appended to the session's growing conversation,
    not the whole history resent every time. DELETE /v1/sessions/{id} frees a
    session explicitly; there is no idle-timeout eviction in this pass.

    Operational endpoints: GET /healthz (always open, no auth -- orchestrators
    must be able to reach it without credentials) and GET /metrics (Prometheus
    text exposition format). If api_key is given, every other route requires
    `Authorization: Bearer <api_key>`; if None (the default), every route
    stays open, exactly like before this parameter existed. A request that
    finds the scheduler already at capacity (SchedulerAtCapacityError) gets
    HTTP 429 with a Retry-After header, an explicit backpressure signal
    instead of queuing indefinitely with no feedback.
    """
    app = FastAPI(title="pager_hf serving (Stage 1: round-robin)")

    async def require_api_key(authorization: str | None = Header(default=None)):
        if api_key is None:
            return
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="missing or invalid Authorization header")

    protected = [Depends(require_api_key)]

    @app.get("/healthz")
    async def healthz():
        status = scheduler.health()
        if not status["thread_alive"]:
            raise HTTPException(status_code=503, detail=status)
        return status

    @app.get("/metrics", dependencies=protected)
    async def metrics():
        return Response(generate_latest(scheduler.metrics_registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/sessions", dependencies=protected)
    async def create_session():
        try:
            session_id = scheduler.create_session()
        except SessionCapacityError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        return {"session_id": session_id}

    @app.delete("/v1/sessions/{session_id}", dependencies=protected)
    async def delete_session(session_id: str):
        try:
            scheduler.delete_session(session_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown session_id {session_id!r}")
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"status": "deleted"}

    @app.post("/v1/completions", dependencies=protected)
    async def completions(request: CompletionRequest):
        if request.max_tokens < 1:
            raise HTTPException(status_code=400, detail="max_tokens must be >= 1")

        encoded = tokenizer(request.prompt, return_tensors="pt")

        try:
            if request.session_id is not None:
                req = scheduler.submit_turn(
                    request.session_id,
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    request.max_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
            else:
                req = scheduler.submit(
                    encoded["input_ids"],
                    encoded["attention_mask"],
                    request.max_tokens,
                    do_sample=request.do_sample,
                    temperature=request.temperature,
                    top_k=request.top_k,
                    top_p=request.top_p,
                )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown session_id {request.session_id!r}")
        except SessionBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except SchedulerAtCapacityError as exc:
            raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": _RETRY_AFTER_SECONDS})

        if request.stream:
            return StreamingResponse(_stream_tokens(req, tokenizer), media_type="text/event-stream")

        await asyncio.to_thread(req.done.wait)
        if req.error is not None:
            raise HTTPException(status_code=500, detail=str(req.error))

        text = tokenizer.decode(req.generated_tokens, skip_special_tokens=True)
        return {"id": req.request_id, "session_id": req.session_id, "choices": [{"text": text}]}

    return app


async def _stream_tokens(req, tokenizer):
    while True:
        token_id = await asyncio.to_thread(req.output_queue.get)
        if token_id is None:
            break
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        yield f"data: {json.dumps({'token': piece})}\n\n"

    if req.error is not None:
        yield f"data: {json.dumps({'error': str(req.error)})}\n\n"
    yield "data: [DONE]\n\n"
