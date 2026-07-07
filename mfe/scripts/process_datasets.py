#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将下载的数据集处理为 client 可用的格式，透传原数据集的全部字段。"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List


def _to_json_safe(obj: Any) -> Any:
    """将 numpy 等类型转为 JSON 可序列化格式。"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def process_drop(data_dir: str, limit: int | None) -> List[Dict[str, Any]]:
    import pandas as pd
    path = os.path.join(data_dir, "drop", "drop.parquet")
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        row_d = _to_json_safe(row.to_dict())
        passage = row_d.get("passage", "") or ""
        question_short = row_d.get("question", "") or ""
        row_d["question_short"] = question_short
        row_d["question"] = f"Passage:\n{passage}\n\nQuestion:\n{question_short}"
        out.append(row_d)
        if limit and len(out) >= limit:
            break
    return out


def process_hotpotqa(data_dir: str, limit: int | None) -> List[Dict[str, Any]]:
    import pandas as pd
    path = os.path.join(data_dir, "hotpotqa", "hotpotqa.parquet")
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        row_d = _to_json_safe(row.to_dict())
        question_short = row_d.get("question", "") or ""
        row_d["question_short"] = question_short
        ctx = row_d.get("context", "")
        if isinstance(ctx, list):
            parts = []
            for item in ctx:
                if isinstance(item, dict):
                    title = item.get("title", "")
                    sents = item.get("sentences", [])
                    text = " ".join(str(s) for s in sents) if isinstance(sents, list) else str(sents)
                    parts.append(f"[{title}]\n{text}")
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    title, sents = item[0], item[1]
                    text = " ".join(str(s) for s in sents) if isinstance(sents, list) else str(sents)
                    parts.append(f"[{title}]\n{text}")
                else:
                    parts.append(str(item))
            ctx_str = "\n\n".join(parts)
        else:
            ctx_str = str(ctx) if ctx else ""
        row_d["question"] = f"Context:\n{ctx_str}\n\nQuestion:\n{question_short}" if ctx_str else question_short
        out.append(row_d)
        if limit and len(out) >= limit:
            break
    return out


def process_math(data_dir: str, limit: int | None) -> List[Dict[str, Any]]:
    import pandas as pd
    path = os.path.join(data_dir, "math", "math.parquet")
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        row_d = _to_json_safe(row.to_dict())
        row_d["question"] = row_d.get("problem", "") or ""
        out.append(row_d)
        if limit and len(out) >= limit:
            break
    return out


def process_gsm8k(data_dir: str, limit: int | None) -> List[Dict[str, Any]]:
    import pandas as pd
    path = os.path.join(data_dir, "gsm8k", "gsm8k.parquet")
    if not os.path.isfile(path):
        return []
    df = pd.read_parquet(path)
    out = []
    for _, row in df.iterrows():
        row_d = _to_json_safe(row.to_dict())
        row_d["question"] = str(row_d.get("question", "") or "").strip()
        out.append(row_d)
        if limit and len(out) >= limit:
            break
    return out


PROCESSORS = {
    "drop": process_drop,
    "hotpotqa": process_hotpotqa,
    "math": process_math,
    "gsm8k": process_gsm8k,
}


def main() -> None:
    p = argparse.ArgumentParser(description="将数据集处理为 client 可用的 questions.json")
    p.add_argument("--data-dir", default="data", help="数据目录（含 drop/hotpotqa/math/gsm8k）")
    p.add_argument("--datasets", nargs="+", default=["drop", "hotpotqa", "math", "gsm8k"])
    p.add_argument("--limit", type=int, default=None, help="每个数据集最多取 N 条")
    p.add_argument("--yaml", default="adv_reason_3.yaml", help="所有题目使用的 yaml 模板")
    p.add_argument("-o", "--output-dir", default=None, help="输出目录，默认与 data-dir 相同")
    args = p.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    out_dir = os.path.abspath(args.output_dir or args.data_dir)
    yaml_name = args.yaml if args.yaml.endswith(".yaml") else f"{args.yaml}.yaml"

    for name in args.datasets:
        name = name.lower()
        if name not in PROCESSORS:
            print(f"Unknown dataset: {name}, skip.")
            continue
        rows = PROCESSORS[name](data_dir, args.limit)
        if not rows:
            print(f"{name}: no data, skip.")
            continue
        for r in rows:
            r["yaml"] = yaml_name
        out_path = os.path.join(out_dir, f"questions_{name}.json")
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"{name}: {len(rows)} items -> {out_path}")


if __name__ == "__main__":
    main()
