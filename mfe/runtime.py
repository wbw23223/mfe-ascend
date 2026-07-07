"""Runtime configuration helpers for deployment-friendly CLI entrypoints."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def truthy(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return value.strip().lower() in TRUE_VALUES


def env_or_default(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def resolve_accelerator(accelerator: str | None = None) -> str:
    value = (accelerator or os.environ.get("MFE_ACCELERATOR", "ascend")).strip().lower()
    if value not in ("ascend", "cuda"):
        raise ValueError("accelerator must be one of: ascend, cuda")
    return value


def apply_offline_env(enabled: bool) -> None:
    if not enabled:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


def apply_device_env(device_ids: str | None, accelerator: str | None = None) -> None:
    backend = resolve_accelerator(accelerator)
    if backend == "ascend":
        os.environ.setdefault("VLLM_TARGET_DEVICE", "npu")
        os.environ.setdefault("VLLM_USE_V1", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if device_ids:
        os.environ["MFE_DEVICE_IDS"] = device_ids
        if backend == "ascend":
            os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
            os.environ["NPU_VISIBLE_DEVICES"] = device_ids
        elif backend == "cuda":
            os.environ["CUDA_VISIBLE_DEVICES"] = device_ids


def package_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(cwd: str | None = None) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def split_device_ids(value: str | None) -> list[int]:
    if not value:
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


@dataclass(frozen=True)
class RuntimeConfig:
    accelerator: str
    device_ids: str | None
    model_path: str | None
    data_dir: str
    output_dir: str | None
    offline: bool

    @classmethod
    def from_values(
        cls,
        *,
        accelerator: str | None = None,
        device_ids: str | None = None,
        model_path: str | None = None,
        data_dir: str | None = None,
        output_dir: str | None = None,
        offline: bool | None = None,
        project_root: str | None = None,
    ) -> "RuntimeConfig":
        root = project_root or os.getcwd()
        resolved_accelerator = accelerator or env_or_default("MFE_ACCELERATOR", "ascend") or "ascend"
        resolved_device_ids = device_ids or env_or_default("MFE_DEVICE_IDS")
        resolved_model_path = model_path or env_or_default("MFE_MODEL_PATH")
        resolved_data_dir = data_dir or env_or_default("MFE_DATA_DIR") or os.path.join(root, "data")
        resolved_output_dir = output_dir or env_or_default("MFE_OUTPUT_DIR")
        resolved_offline = truthy(os.environ.get("MFE_OFFLINE")) if offline is None else offline
        return cls(
            accelerator=resolved_accelerator,
            device_ids=resolved_device_ids,
            model_path=resolved_model_path,
            data_dir=os.path.abspath(resolved_data_dir),
            output_dir=os.path.abspath(resolved_output_dir) if resolved_output_dir else None,
            offline=resolved_offline,
        )

    def apply(self) -> None:
        resolved_accelerator = resolve_accelerator(self.accelerator)
        os.environ["MFE_ACCELERATOR"] = resolved_accelerator
        if self.model_path:
            os.environ["MFE_MODEL_PATH"] = self.model_path
        os.environ["MFE_DATA_DIR"] = self.data_dir
        if self.output_dir:
            os.environ["MFE_OUTPUT_DIR"] = self.output_dir
        if self.offline:
            os.environ["MFE_OFFLINE"] = "1"
        apply_device_env(self.device_ids, resolved_accelerator)
        apply_offline_env(self.offline)


def collect_run_info(config: RuntimeConfig | None = None, cwd: str | None = None) -> dict[str, Any]:
    config = config or RuntimeConfig.from_values(project_root=cwd)
    resolved_accelerator = resolve_accelerator(config.accelerator)
    return {
        "git_commit": git_commit(cwd),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "accelerator": resolved_accelerator,
        "requested_accelerator": config.accelerator,
        "device_ids": config.device_ids,
        "model_path": config.model_path,
        "data_dir": config.data_dir,
        "output_dir": config.output_dir,
        "offline": config.offline,
        "max_model_len": os.environ.get("MFE_MAX_MODEL_LEN"),
        "gpu_memory_utilization": os.environ.get("MFE_GPU_MEMORY_UTILIZATION"),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend", "transformers")
        },
    }
