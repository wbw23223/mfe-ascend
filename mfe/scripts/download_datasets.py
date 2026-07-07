#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下载 DROP、HotpotQA、MATH、GSM8K 数据集到 data/ 目录。"""

from __future__ import annotations

import argparse
import os

# 避免 OpenMP libiomp5md 重复加载（与 PyTorch/numpy 冲突）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


from datasets import concatenate_datasets, load_dataset


MATH_CONFIGS = [
    "algebra", "counting_and_probability", "geometry", "intermediate_algebra",
    "number_theory", "prealgebra", "precalculus",
]


DATASET_CONFIGS = {
    "drop": {
        "path": "ucinlp/drop",
        "split": "validation",
        "text_key": "question",
        "passage_key": "passage",
        "answer_key": "answers_spans",
    },
    "hotpotqa": {
        "path": "hotpotqa/hotpot_qa",
        "name": "distractor",
        "split": "validation",
        "text_key": "question",
        "context_key": "context",
        "answer_key": "answer",
    },
    "math": {
        "path": "EleutherAI/hendrycks_math",
        "split": "test",
        "text_key": "problem",
        "answer_key": "solution",
    },
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "text_key": "question",
        "answer_key": "answer",
    },
}


def download_dataset(name: str, data_dir: str, limit: int | None = None) -> str:
    name = name.lower()
    cfg = DATASET_CONFIGS[name]
    out_dir = os.path.join(data_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {name}...")
    if name == "math":
        # MATH 按 subject 分 config，需加载并合并
        parts = []
        for subj in MATH_CONFIGS:
            part = load_dataset(cfg["path"], subj, split=cfg["split"])
            parts.append(part)
        ds = concatenate_datasets(parts)
    else:
        kwargs = {"path": cfg["path"], "split": cfg["split"]}
        if "name" in cfg:
            kwargs["name"] = cfg["name"]
        ds = load_dataset(**kwargs)

    if limit is not None and len(ds) > limit:
        ds = ds.select(range(limit))
    out_path = os.path.join(out_dir, f"{name}.parquet")
    ds.to_parquet(out_path)
    print(f"  Saved to {out_path} ({len(ds)} examples)")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="下载 DROP/HotpotQA/MATH/GSM8K 到 data/")
    p.add_argument("--data-dir", default="data", help="数据输出目录")
    p.add_argument("--datasets", nargs="+", default=["drop", "hotpotqa", "math", "gsm8k"])
    p.add_argument("--limit", type=int, default=None, help="每个数据集最多下载 N 条（测试用）")
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)

    for name in args.datasets:
        name = name.lower()
        if name not in DATASET_CONFIGS:
            print(f"Unknown dataset: {name}, skip.")
            continue
        try:
            download_dataset(name, data_dir, args.limit)
        except Exception as e:
            print(f"  Failed: {e}")


if __name__ == "__main__":
    main()
