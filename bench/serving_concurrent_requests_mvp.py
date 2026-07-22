from __future__ import annotations

import threading
import time

import torch
from config import MODEL_NAME, POLICY, RAM_BUDGET, TOKENS_PER_BLOCK, VRAM_BUDGET
from fastapi.testclient import TestClient
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel
from pager_hf.serving.app import create_app
from pager_hf.serving.scheduler import ContinuousBatchingScheduler

# Real proof of the serving layer, Stage 1 (round-robin) and Stage 2 (batched
# cross-session decode, now wired into this same scheduler) alike, wrapped in
# a FastAPI app. Four things to prove, on a real model:
# 1. Correctness: a request served through the scheduler produces the exact
#    same tokens as calling PagedModel.generate() directly -- the project's
#    standing same_token_ids bar, not just "the HTTP layer doesn't crash."
# 2. Genuine interleaving: a short request submitted alongside a long one
#    finishes without waiting for the long one to fully complete -- the
#    actual point of round-robin scheduling, not just faster-sounding prose.
# 3. Stage 2 actually engages for real concurrent requests (not just admitted
#    one at a time): several same-length requests submitted together should
#    converge to a shared tail length after their first step and get routed
#    through _step_batched -- checked directly, not assumed from the design.
# 4. Multi-turn sessions: two /v1/completions calls sharing one session_id,
#    the second call's prompt a genuinely NEW piece of text (not a repeat),
#    must produce the same output a client would get from one PagedModel
#    instance's generate() called twice in a row -- proving the KV cache
#    really carries over between HTTP calls, not just that the endpoint
#    accepts a session_id without crashing.

# Long enough to clear tokens_per_block worth of full pager blocks before the
# first decode step -- a short prompt (a handful of tokens) hits a separate,
# pre-existing bug in attention-scoring policies unrelated to this serving
# layer (tracked separately), so this avoids that edge case rather than
# working around it here.
PROMPT = "The quick brown fox jumps over the lazy dog. " * 12 + "Once upon a time,"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    # ---- Correctness: scheduler-served tokens vs a direct PagedModel.generate() call ----
    print("\nChecking token-exact correctness (scheduler vs direct PagedModel.generate())...")
    encoded = tokenizer(PROMPT, return_tensors="pt").to(device)

    direct_model = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy=POLICY)
    direct_ids = direct_model.generate(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], max_new_tokens=8
    )

    scheduler = ContinuousBatchingScheduler(
        model,
        tokenizer,
        max_concurrent_sessions=2,
        total_vram_budget=VRAM_BUDGET * 2,
        total_ram_budget=RAM_BUDGET * 2,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
        device=device,
    )
    scheduler.start()

    scheduled_req = scheduler.submit(encoded["input_ids"], encoded["attention_mask"], max_new_tokens=8)
    scheduled_req.done.wait(timeout=60)

    print("direct_ids:   ", direct_ids)
    print("scheduled_ids:", scheduled_req.generated_tokens)
    assert scheduled_req.generated_tokens == direct_ids, "scheduler-served generation diverged from direct generate()"
    print("OK: scheduler-served tokens match direct PagedModel.generate() exactly.")

    # ---- Genuine interleaving, over real HTTP, via TestClient ----
    print("\nChecking genuine interleaving (long request + short request, both via HTTP)...")
    app = create_app(scheduler, tokenizer)
    client = TestClient(app)

    results: dict[str, float] = {}

    def call(name: str, max_tokens: int) -> None:
        client.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": max_tokens})
        results[name] = time.perf_counter()

    long_thread = threading.Thread(target=call, args=("long", 40))
    short_thread = threading.Thread(target=call, args=("short", 3))
    started = time.perf_counter()
    long_thread.start()
    time.sleep(0.05)  # let the long request get admitted first
    short_thread.start()
    long_thread.join(timeout=120)
    short_thread.join(timeout=120)

    short_elapsed = results["short"] - started
    long_elapsed = results["long"] - started
    print(f"short_completed_at_s: {short_elapsed:.3f}")
    print(f"long_completed_at_s:  {long_elapsed:.3f}")
    assert short_elapsed < long_elapsed, "short request did not finish before the long one -- not interleaving"
    print("OK: short request completed without waiting for the long one to fully finish.")

    scheduler.stop()
    print("\nOK: serving layer verified -- token-exact correctness and genuine round-robin interleaving both hold.")

    # ---- Stage 2 actually engages for real concurrent requests submitted together ----
    print("\nChecking Stage 2 (batched cross-session decode) engages for real concurrent requests...")
    batched_scheduler = ContinuousBatchingScheduler(
        model,
        tokenizer,
        max_concurrent_sessions=3,
        total_vram_budget=VRAM_BUDGET * 3,
        total_ram_budget=RAM_BUDGET * 3,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
        device=device,
    )

    batched_call_count = 0
    original_step_batched = batched_scheduler._step_batched

    def counting_step_batched(group):
        nonlocal batched_call_count
        batched_call_count += 1
        original_step_batched(group)

    batched_scheduler._step_batched = counting_step_batched
    batched_scheduler.start()

    requests = [
        batched_scheduler.submit(encoded["input_ids"], encoded["attention_mask"], max_new_tokens=8) for _ in range(3)
    ]
    for req in requests:
        req.done.wait(timeout=60)

    all_match = all(req.generated_tokens == direct_ids for req in requests)
    print("concurrent_requests_ids:", [req.generated_tokens for req in requests])
    print("batched_step_calls:", batched_call_count)
    assert all_match, "at least one concurrently-served request diverged from direct PagedModel.generate()"
    assert batched_call_count > 0, "Stage 2's batched decode path never engaged for 3 same-length concurrent requests"
    print("OK: 3 concurrent same-length requests all match direct generate() exactly, via the real batched path.")

    batched_scheduler.stop()
    print("\nOK: serving layer fully verified -- correctness, interleaving, and real Stage 2 batching all hold.")

    # ---- Multi-turn sessions: KV cache really carries over between HTTP calls ----
    print("\nChecking multi-turn sessions (session_id reused across two /v1/completions calls)...")
    SECOND_TURN_TEXT = " Continuing the story, the fox then"

    reference_session = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy=POLICY)
    reference_turn_1_ids = reference_session.generate(
        input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"], max_new_tokens=4
    )
    reference_turn_1_text = tokenizer.decode(reference_turn_1_ids, skip_special_tokens=True)

    full_ids_after_turn_1 = torch.cat(
        [encoded["input_ids"], torch.tensor([reference_turn_1_ids], device=device)], dim=1
    )
    second_turn_encoded = tokenizer(SECOND_TURN_TEXT, return_tensors="pt").to(device)
    reference_turn_2_input_ids = torch.cat([full_ids_after_turn_1, second_turn_encoded["input_ids"]], dim=1)
    reference_turn_2_attention_mask = torch.ones_like(reference_turn_2_input_ids)
    reference_turn_2_ids = reference_session.generate(
        input_ids=reference_turn_2_input_ids, attention_mask=reference_turn_2_attention_mask, max_new_tokens=4
    )
    reference_turn_2_text = tokenizer.decode(reference_turn_2_ids, skip_special_tokens=True)

    session_scheduler = ContinuousBatchingScheduler(
        model,
        tokenizer,
        max_concurrent_sessions=2,
        total_vram_budget=VRAM_BUDGET * 2,
        total_ram_budget=RAM_BUDGET * 2,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
        device=device,
    )
    session_scheduler.start()
    session_app = create_app(session_scheduler, tokenizer)
    session_client = TestClient(session_app)

    session_id = session_client.post("/v1/sessions").json()["session_id"]
    turn_1_response = session_client.post(
        "/v1/completions", json={"prompt": PROMPT, "max_tokens": 4, "session_id": session_id}
    ).json()
    turn_2_response = session_client.post(
        "/v1/completions", json={"prompt": SECOND_TURN_TEXT, "max_tokens": 4, "session_id": session_id}
    ).json()

    print("reference_turn_1_text:", repr(reference_turn_1_text))
    print("served_turn_1_text:   ", repr(turn_1_response["choices"][0]["text"]))
    print("reference_turn_2_text:", repr(reference_turn_2_text))
    print("served_turn_2_text:   ", repr(turn_2_response["choices"][0]["text"]))
    assert turn_1_response["session_id"] == session_id
    assert turn_2_response["session_id"] == session_id
    assert turn_1_response["choices"][0]["text"] == reference_turn_1_text, "turn 1 diverged from the reference session"
    assert (
        turn_2_response["choices"][0]["text"] == reference_turn_2_text
    ), "turn 2 diverged -- the served session's KV cache did not carry over correctly between HTTP calls"
    print("OK: both turns of a served session match a direct PagedModel session generating the same two turns.")

    delete_status = session_client.delete(f"/v1/sessions/{session_id}").json()
    assert delete_status["status"] == "deleted"
    assert session_client.delete(f"/v1/sessions/{session_id}").status_code == 404
    print("OK: DELETE /v1/sessions/{id} frees a session and 404s on a second delete.")

    # ---- Metrics: /metrics reflects the real traffic already generated above ----
    print("\nChecking /metrics reflects real request traffic (not just unit-tested against fakes)...")
    metrics_text = session_client.get("/metrics").text
    metrics = {}
    for line in metrics_text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        metrics[name.split("{")[0]] = metrics.get(name.split("{")[0], 0.0) + float(value)

    print("pager_hf_requests_total:", metrics.get("pager_hf_requests_total"))
    print("pager_hf_tokens_generated_total:", metrics.get("pager_hf_tokens_generated_total"))
    print("pager_hf_step_calls_total:", metrics.get("pager_hf_step_calls_total"))
    assert metrics.get("pager_hf_requests_total", 0) >= 2, "expected at least the 2 session turns above to be counted"
    assert (
        metrics.get("pager_hf_tokens_generated_total", 0) >= 8
    ), "expected at least the 8 tokens generated across both turns above to be counted"
    assert metrics.get("pager_hf_step_calls_total", 0) > 0, "expected some decode-step calls to be counted"
    print("OK: /metrics reflects the real request traffic generated in this run.")

    session_scheduler.stop()
    print(
        "\nOK: serving layer fully verified -- correctness, interleaving, Stage 2 batching, multi-turn sessions, "
        "and metrics all hold."
    )


if __name__ == "__main__":
    main()
