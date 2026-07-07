#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a minimal vLLM generation to verify the Ascend backend before MFE."""

from __future__ import annotations

import argparse
import os

from mfe.runtime import RuntimeConfig


def main() -> None:
    p = argparse.ArgumentParser(description="vLLM Ascend smoke test")
    p.add_argument("--model-path", required=True, help="本地模型目录")
    p.add_argument("--prompt", default="What is 1+1? Answer briefly.", help="测试 prompt")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=None)
    p.add_argument("--device-ids", default=None, help="设备 ID，如 0；也可用 MFE_DEVICE_IDS")
    p.add_argument("--accelerator", default="ascend", choices=("ascend", "cuda"))
    p.add_argument("--offline", action="store_true")
    args = p.parse_args()

    cfg = RuntimeConfig.from_values(
        accelerator=args.accelerator,
        device_ids=args.device_ids,
        model_path=args.model_path,
        offline=args.offline or None,
    )
    cfg.apply()

    if not os.path.isdir(args.model_path):
        raise SystemExit(f"model path does not exist or is not a directory: {args.model_path}")

    from vllm import LLM, SamplingParams

    gpu_memory_utilization = args.gpu_memory_utilization
    if gpu_memory_utilization is None and os.environ.get("MFE_GPU_MEMORY_UTILIZATION"):
        gpu_memory_utilization = float(os.environ["MFE_GPU_MEMORY_UTILIZATION"])
    max_model_len = args.max_model_len
    if max_model_len is None and os.environ.get("MFE_MAX_MODEL_LEN"):
        max_model_len = int(os.environ["MFE_MAX_MODEL_LEN"])
    if max_model_len is None:
        max_model_len = 2048
    llm_kwargs = {
        "model": args.model_path,
        "dtype": args.dtype,
        "max_model_len": max_model_len,
        "enforce_eager": True,
    }
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization
    llm = LLM(**llm_kwargs)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    outputs = llm.generate([args.prompt], params)
    text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
    print(text.strip())


if __name__ == "__main__":
    main()
