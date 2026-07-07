#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查 openEuler/Ascend/vLLM Ascend 运行环境。"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from mfe.runtime import RuntimeConfig, split_device_ids


def _run(cmd: list[str], timeout: int = 10) -> dict[str, Any]:
    if shutil.which(cmd[0]) is None:
        return {"ok": False, "error": f"{cmd[0]} not found"}
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout, check=False)
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _package_version(dist_name: str, import_name: str | None = None) -> dict[str, Any]:
    import_name = import_name or dist_name.replace("-", "_")
    result: dict[str, Any] = {"installed": False}
    try:
        result["version"] = importlib.metadata.version(dist_name)
        result["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        result["version"] = None
    try:
        importlib.import_module(import_name)
        result["importable"] = True
    except Exception as exc:
        result["importable"] = False
        result["import_error"] = repr(exc)
    return result


def _os_release() -> dict[str, str]:
    path = "/etc/os-release"
    if not os.path.exists(path):
        return {}
    data: dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.rstrip("\n").split("=", 1)
            data[key] = value.strip('"')
    return data


def _torch_npu_summary() -> dict[str, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}

    summary: dict[str, Any] = {"ok": True}
    try:
        summary["device_count"] = int(torch.npu.device_count())
        summary["is_available"] = bool(torch.npu.is_available())
    except Exception as exc:
        summary["device_error"] = repr(exc)
    return summary


def _path_check(path: str | None, kind: str) -> dict[str, Any]:
    if not path:
        return {"ok": False, "error": f"{kind} path is not set"}
    exists = os.path.exists(path)
    return {"ok": exists, "path": path, "error": None if exists else f"{kind} path does not exist"}


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Check MFE Ascend runtime environment")
    p.add_argument("--model-path", default=None, help="本地模型目录；也可用 MFE_MODEL_PATH")
    p.add_argument("--data-dir", default=None, help="数据目录；也可用 MFE_DATA_DIR")
    p.add_argument("--device-ids", default=None, help="设备 ID，如 0 或 0,1；也可用 MFE_DEVICE_IDS")
    p.add_argument("--accelerator", default=None, choices=("ascend", "cuda"), help="推理后端")
    p.add_argument("--offline", action="store_true", help="启用离线模式环境变量")
    args = p.parse_args()

    cfg = RuntimeConfig.from_values(
        accelerator=args.accelerator,
        device_ids=args.device_ids,
        model_path=args.model_path,
        data_dir=args.data_dir,
        offline=args.offline or None,
    )
    cfg.apply()
    packages = {
        "torch": _package_version("torch"),
        "torch-npu": _package_version("torch-npu", "torch_npu"),
        "vllm": _package_version("vllm"),
        "vllm-ascend": _package_version("vllm-ascend", "vllm_ascend"),
        "transformers": _package_version("transformers"),
    }
    env = {
        key: os.environ.get(key)
        for key in (
            "MFE_ACCELERATOR",
            "MFE_DEVICE_IDS",
            "MFE_MODEL_PATH",
            "MFE_DATA_DIR",
            "MFE_OFFLINE",
            "MFE_MAX_MODEL_LEN",
            "MFE_GPU_MEMORY_UTILIZATION",
            "ASCEND_HOME_PATH",
            "ASCEND_RT_VISIBLE_DEVICES",
            "NPU_VISIBLE_DEVICES",
            "VLLM_TARGET_DEVICE",
            "VLLM_USE_V1",
            "VLLM_WORKER_MULTIPROC_METHOD",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "HF_DATASETS_OFFLINE",
            "LD_LIBRARY_PATH",
        )
    }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": _os_release(),
        "env": env,
        "runtime": {
            "accelerator": cfg.accelerator,
            "device_ids": cfg.device_ids,
            "parsed_device_ids": split_device_ids(cfg.device_ids),
            "model_path": cfg.model_path,
            "data_dir": cfg.data_dir,
            "offline": cfg.offline,
        },
        "paths": {
            "model_path": _path_check(cfg.model_path, "model"),
            "data_dir": _path_check(cfg.data_dir, "data"),
            "gsm8k_parquet": _path_check(os.path.join(cfg.data_dir, "gsm8k", "gsm8k.parquet"), "gsm8k parquet"),
        },
        "commands": {
            "npu-smi info": _run(["npu-smi", "info"]),
            "uname -a": _run(["uname", "-a"]),
            "gcc --version": _run(["gcc", "--version"]),
        },
        "packages": packages,
        "torch_npu": _torch_npu_summary(),
    }
    failures = []
    if cfg.accelerator == "ascend" and not report["commands"]["npu-smi info"].get("ok"):
        failures.append("npu-smi info failed")
    if cfg.accelerator == "ascend" and not report["packages"]["torch-npu"].get("importable"):
        failures.append("torch_npu is not importable")
    if cfg.accelerator == "ascend" and not report["packages"]["vllm-ascend"].get("importable"):
        failures.append("vllm_ascend is not importable")
    if cfg.accelerator == "ascend" and int(report["torch_npu"].get("device_count") or 0) <= 0:
        failures.append("torch.npu.device_count() is 0")
    if not report["paths"]["model_path"]["ok"]:
        failures.append(report["paths"]["model_path"]["error"])
    if not report["paths"]["data_dir"]["ok"]:
        failures.append(report["paths"]["data_dir"]["error"])
    if not report["paths"]["gsm8k_parquet"]["ok"]:
        failures.append(report["paths"]["gsm8k_parquet"]["error"])
    report["summary"] = {"ok": not failures, "failures": failures}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
