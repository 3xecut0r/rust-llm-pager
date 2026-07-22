from __future__ import annotations

import argparse
import logging
import os

import torch
import uvicorn
from prometheus_client import CollectorRegistry
from transformers import AutoModelForCausalLM, AutoTokenizer

from .app import create_app
from .multi_gpu import MultiGpuScheduler
from .scheduler import ContinuousBatchingScheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="pager_hf serving layer (Stage 1: round-robin continuous batching)")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--policy", default="sinks_heavy_hitter")
    parser.add_argument("--tokens-per-block", type=int, default=16)
    parser.add_argument("--no-streaming-attention", action="store_true", help="disable use_streaming_attention")
    parser.add_argument("--streaming-group-size-blocks", type=int, default=64)
    parser.add_argument("--max-concurrent-sessions", type=int, default=4, help="per GPU, not shared across --devices")
    parser.add_argument("--max-queue-depth", type=int, default=100, help="per GPU; see --help for details")
    parser.add_argument("--total-vram-budget-mb", type=int, required=True, help="per GPU, not split across --devices")
    parser.add_argument("--total-ram-budget-mb", type=int, required=True, help="per GPU, not split across --devices")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--devices",
        default="cuda:0",
        help="comma-separated CUDA devices, one full model replica + scheduler per device "
        "(data-parallel, not model parallelism -- each replica must fit the model on its own). "
        "Requests load-balance across them by current load; a multi-turn session stays on "
        "whichever device it started on for its whole lifetime.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="if set (or PAGER_HF_API_KEY is set in the environment), require "
        "'Authorization: Bearer <key>' on every route except /healthz. Open by default.",
    )
    parser.add_argument(
        "--shutdown-timeout",
        type=float,
        default=30.0,
        help="seconds to let in-flight requests finish on SIGTERM/SIGINT before forcing them to fail",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to serve.")
    devices = [torch.device(d.strip()) for d in args.devices.split(",")]

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # One shared registry only when there's more than one device -- a single
    # scheduler keeps its own private registry, unchanged from before this
    # multi-GPU option existed.
    shared_registry = CollectorRegistry() if len(devices) > 1 else None

    schedulers = []
    for device in devices:
        model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", torch_dtype=dtype).to(
            device
        )
        model.eval()

        schedulers.append(
            ContinuousBatchingScheduler(
                model,
                tokenizer,
                max_concurrent_sessions=args.max_concurrent_sessions,
                max_queue_depth=args.max_queue_depth,
                total_vram_budget=args.total_vram_budget_mb * 1_000_000,
                total_ram_budget=args.total_ram_budget_mb * 1_000_000,
                policy=args.policy,
                tokens_per_block=args.tokens_per_block,
                use_streaming_attention=not args.no_streaming_attention,
                streaming_group_size_blocks=args.streaming_group_size_blocks,
                device=device,
                metrics_registry=shared_registry,
                device_label=str(device),
            )
        )

    scheduler = MultiGpuScheduler(schedulers) if len(schedulers) > 1 else schedulers[0]
    scheduler.start()

    api_key = args.api_key or os.environ.get("PAGER_HF_API_KEY")
    app = create_app(scheduler, tokenizer, api_key=api_key)
    try:
        # uvicorn's own SIGTERM/SIGINT handling stops accepting new HTTP
        # connections and returns once its own graceful wind-down is done --
        # no custom signal.signal() registration needed. The scheduler's own
        # in-flight decode work is drained separately, below, once uvicorn
        # has actually returned.
        uvicorn.run(app, host=args.host, port=args.port, timeout_graceful_shutdown=int(args.shutdown_timeout))
    finally:
        scheduler.stop(timeout=args.shutdown_timeout)


if __name__ == "__main__":
    main()
