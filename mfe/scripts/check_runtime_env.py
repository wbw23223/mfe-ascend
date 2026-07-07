#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check a CUDA or Ascend runtime before running MFE."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any

from mfe.runtime import RuntimeConfig, resolve_accelerator, split_device_ids
from mfe.util import visible_accelerator_device_ids


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


def _package_status(dist_name: str, import_name: str | None = None) -> dict[str, Any]:
    import_name = import_name or dist_name.replace("-", "_")
    result: dict[str, Any] = {"installed": False, "importable": False, "version": None}
    try:
        result["version"] = importlib.metadata.version(dist_name)
        result["installed"] = True
    except importlib.metadata.PackageNotFoundError:
        pass
    try:
        importlib.import_module(import_name)
        result["importable"] = True
    except Exception as exc:
        result["import_error"] = repr(exc)
    return result


def _int_or_none(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _path_check(path: str | None, kind: str) -> dict[str, Any]:
    if not path:
        return {"ok": False, "path": path, "error": f"{kind} path is not set"}
    exists = os.path.exists(path)
    return {"ok": exists, "path": path, "error": None if exists else f"{kind} path does not exist"}


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


def _nvidia_smi_inventory() -> dict[str, Any]:
    command = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,uuid",
            "--format=csv,noheader,nounits",
        ]
    )
    devices: list[dict[str, Any]] = []
    if command.get("ok"):
        for line in command.get("stdout", "").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 4:
                continue
            devices.append(
                {
                    "index": _int_or_none(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": _int_or_none(parts[2]),
                    "uuid": parts[3],
                }
            )
    return {"command": command, "devices": devices}


def _torch_cuda_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "devices": []}

    summary: dict[str, Any] = {"ok": True, "devices": []}
    try:
        summary["is_available"] = bool(torch.cuda.is_available())
        summary["device_count"] = int(torch.cuda.device_count())
        for logical_id in range(int(summary["device_count"])):
            props = torch.cuda.get_device_properties(logical_id)
            summary["devices"].append(
                {
                    "logical_id": logical_id,
                    "name": torch.cuda.get_device_name(logical_id),
                    "memory_total_mb": int(getattr(props, "total_memory", 0) // (1024 * 1024)),
                    "capability": ".".join(str(x) for x in torch.cuda.get_device_capability(logical_id)),
                }
            )
    except Exception as exc:
        summary["device_error"] = repr(exc)
    return summary


def _torch_npu_summary() -> dict[str, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "devices": []}

    summary: dict[str, Any] = {"ok": True, "devices": []}
    try:
        count = int(torch.npu.device_count())
        summary["is_available"] = bool(torch.npu.is_available())
        summary["device_count"] = count
        for logical_id in range(count):
            name = None
            memory_total_mb = None
            if hasattr(torch.npu, "get_device_name"):
                name = torch.npu.get_device_name(logical_id)
            if hasattr(torch.npu, "get_device_properties"):
                props = torch.npu.get_device_properties(logical_id)
                raw_memory = getattr(props, "total_memory", None) or getattr(props, "totalGlobalMem", None)
                if raw_memory:
                    memory_total_mb = int(raw_memory // (1024 * 1024))
            summary["devices"].append(
                {
                    "logical_id": logical_id,
                    "name": name,
                    "memory_total_mb": memory_total_mb,
                }
            )
    except Exception as exc:
        summary["device_error"] = repr(exc)
    return summary


def _homogeneity_failures(devices: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    if len(devices) < 2:
        return failures
    for key, label in (("name", "device model"), ("memory_total_mb", "device memory")):
        values = {device.get(key) for device in devices if device.get(key) not in (None, "")}
        if len(values) > 1:
            failures.append(f"visible {label} is not homogeneous: {sorted(values)}")
    return failures


def _runtime_devices(backend: str, cuda: dict[str, Any], npu: dict[str, Any]) -> list[dict[str, Any]]:
    if backend == "ascend":
        return list(npu.get("devices") or [])
    return list(cuda.get("devices") or [])


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    cfg = RuntimeConfig.from_values(
        accelerator=args.accelerator,
        device_ids=args.device_ids,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        offline=args.offline or None,
    )
    cfg.apply()
    backend = resolve_accelerator(os.environ.get("MFE_ACCELERATOR"))
    visible_ids = visible_accelerator_device_ids(backend)  # type: ignore[arg-type]
    packages = {
        "torch": _package_status("torch"),
        "torch-npu": _package_status("torch-npu", "torch_npu"),
        "vllm": _package_status("vllm"),
        "vllm-ascend": _package_status("vllm-ascend", "vllm_ascend"),
        "transformers": _package_status("transformers"),
    }
    cuda = _torch_cuda_summary() if backend == "cuda" else {"devices": []}
    npu = _torch_npu_summary() if backend == "ascend" else {"devices": []}
    runtime_devices = _runtime_devices(backend, cuda, npu)
    runtime_device_count = int((npu if backend == "ascend" else cuda).get("device_count") or 0)
    paths = {
        "model_path": _path_check(cfg.model_path, "model"),
        "data_dir": _path_check(cfg.data_dir, "data"),
        "gsm8k_parquet": _path_check(os.path.join(cfg.data_dir, "gsm8k", "gsm8k.parquet"), "gsm8k parquet"),
    }

    failures: list[str] = []
    if backend == "cuda":
        if not packages["torch"].get("importable"):
            failures.append("torch is not importable")
        if not packages["vllm"].get("importable"):
            failures.append("vllm is not importable")
        if int(cuda.get("device_count") or 0) <= 0:
            failures.append("torch.cuda.device_count() is 0")
    else:
        if not packages["torch"].get("importable"):
            failures.append("torch is not importable")
        if not packages["torch-npu"].get("importable"):
            failures.append("torch_npu is not importable")
        if not packages["vllm"].get("importable"):
            failures.append("vllm is not importable")
        if not packages["vllm-ascend"].get("importable"):
            failures.append("vllm_ascend is not importable")
        if int(npu.get("device_count") or 0) <= 0:
            failures.append("torch.npu.device_count() is 0")

    if args.expected_device_count is not None and len(visible_ids) != args.expected_device_count:
        failures.append(
            f"visible device count is {len(visible_ids)}, expected {args.expected_device_count}"
        )
    if args.expected_device_count is not None and runtime_device_count != args.expected_device_count:
        failures.append(
            f"runtime device count is {runtime_device_count}, expected {args.expected_device_count}"
        )
    if args.require_homogeneous:
        failures.extend(_homogeneity_failures(runtime_devices))
    if args.require_model_path and not paths["model_path"]["ok"]:
        failures.append(paths["model_path"]["error"])
    if args.require_data_dir and not paths["data_dir"]["ok"]:
        failures.append(paths["data_dir"]["error"])
    if args.require_gsm8k and not paths["gsm8k_parquet"]["ok"]:
        failures.append(paths["gsm8k_parquet"]["error"])

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "os_release": _os_release(),
        "runtime": {
            "requested_accelerator": args.accelerator or os.environ.get("MFE_ACCELERATOR"),
            "accelerator": backend,
            "device_ids": cfg.device_ids,
            "parsed_device_ids": split_device_ids(cfg.device_ids),
            "visible_device_ids": visible_ids,
            "runtime_device_count": runtime_device_count,
            "expected_device_count": args.expected_device_count,
            "require_homogeneous": args.require_homogeneous,
            "model_path": cfg.model_path,
            "data_dir": cfg.data_dir,
            "output_dir": cfg.output_dir,
            "offline": cfg.offline,
        },
        "env": {
            key: os.environ.get(key)
            for key in (
                "MFE_ACCELERATOR",
                "MFE_DEVICE_IDS",
                "MFE_MODEL_PATH",
                "MFE_DATA_DIR",
                "MFE_OUTPUT_DIR",
                "MFE_OFFLINE",
                "MFE_MAX_MODEL_LEN",
                "MFE_GPU_MEMORY_UTILIZATION",
                "CUDA_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "NPU_VISIBLE_DEVICES",
                "VLLM_TARGET_DEVICE",
                "VLLM_USE_V1",
                "VLLM_WORKER_MULTIPROC_METHOD",
                "LD_LIBRARY_PATH",
            )
        },
        "paths": paths,
        "commands": {
            "nvidia-smi": _run(["nvidia-smi"]) if backend == "cuda" else None,
            "npu-smi info": _run(["npu-smi", "info"]) if backend == "ascend" else None,
            "uname -a": _run(["uname", "-a"]),
        },
        "inventory": {
            "nvidia_smi": _nvidia_smi_inventory() if backend == "cuda" else None,
            "torch_cuda": cuda,
            "torch_npu": npu,
            "runtime_devices": runtime_devices,
        },
        "packages": packages,
        "summary": {"ok": not failures, "failures": failures},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Check MFE CUDA/Ascend runtime environment")
    p.add_argument("--accelerator", default=None, choices=("ascend", "cuda"))
    p.add_argument("--device-ids", default=None, help="visible device IDs, for example 0 or 0,1")
    p.add_argument("--expected-device-count", type=int, default=None)
    p.add_argument("--require-homogeneous", action="store_true")
    p.add_argument("--model-path", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--require-model-path", action="store_true")
    p.add_argument("--require-data-dir", action="store_true")
    p.add_argument("--require-gsm8k", action="store_true")
    args = p.parse_args()

    report = build_report(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["summary"]["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
