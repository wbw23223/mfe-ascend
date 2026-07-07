"""工具：dtype 解析、Ascend/CUDA 设备发现与 worker 设备绑定。"""

from __future__ import annotations

import os
from typing import Iterable, Literal

from mfe.runtime import resolve_accelerator

AcceleratorBackend = Literal["ascend", "cuda"]


def get_accelerator_backend() -> AcceleratorBackend:
    """返回当前推理后端。mfe-ascend 默认使用 Ascend，也可通过 MFE_ACCELERATOR 覆盖。"""
    return resolve_accelerator()  # type: ignore[return-value]


def _parse_visible_ids(names: Iterable[str]) -> list[int] | None:
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        ids: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if part:
                ids.append(int(part))
        return ids
    return None


def _torch_device_count(backend: AcceleratorBackend) -> int:
    if backend == "ascend":
        import torch
        import torch_npu  # noqa: F401

        return int(torch.npu.device_count())

    import torch

    return int(torch.cuda.device_count())


def visible_accelerator_device_ids(backend: AcceleratorBackend | None = None) -> list[int]:
    """返回对当前进程可见的物理设备 ID。"""
    backend = backend or get_accelerator_backend()
    explicit = _parse_visible_ids(("MFE_DEVICE_IDS",))
    if explicit is not None:
        return explicit
    if backend == "ascend":
        visible = _parse_visible_ids(("ASCEND_RT_VISIBLE_DEVICES", "NPU_VISIBLE_DEVICES"))
    else:
        visible = _parse_visible_ids(("CUDA_VISIBLE_DEVICES",))
    if visible is not None:
        return visible

    try:
        return list(range(_torch_device_count(backend)))
    except Exception:
        return []


def configure_worker_device(device_id: int, backend: AcceleratorBackend | None = None) -> None:
    """在 worker 进程内限制 vLLM 只看到一个设备。必须在导入 vLLM 前调用。"""
    backend = backend or get_accelerator_backend()
    if backend == "ascend":
        os.environ["MFE_DEVICE_IDS"] = str(device_id)
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_id)
        os.environ["NPU_VISIBLE_DEVICES"] = str(device_id)
        os.environ.setdefault("VLLM_TARGET_DEVICE", "npu")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # vllm-ascend 0.9.x 推荐 V1 Engine；新版本默认 V1 时该变量也无害。
        os.environ.setdefault("VLLM_USE_V1", "1")
        try:
            import torch
            import torch_npu  # noqa: F401

            if torch.npu.device_count() > 0:
                torch.npu.set_device(0)
        except Exception:
            return
    else:
        os.environ["MFE_DEVICE_IDS"] = str(device_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


def empty_device_cache(backend: AcceleratorBackend | None = None) -> None:
    """清理当前后端显存/NPU HBM cache；失败时不影响主流程退出。"""
    backend = backend or get_accelerator_backend()
    try:
        import torch

        if backend == "ascend":
            import torch_npu  # noqa: F401

            if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()
    except Exception:
        return


def device_label(device_id: int, backend: AcceleratorBackend | None = None) -> str:
    backend = backend or get_accelerator_backend()
    return f"npu:{device_id}" if backend == "ascend" else f"cuda:{device_id}"


def no_visible_devices_message(backend: AcceleratorBackend | None = None) -> str:
    backend = backend or get_accelerator_backend()
    if backend == "ascend":
        return (
            "No visible Ascend NPUs. Check npu-smi info, CANN/torch-npu installation, "
            "and ASCEND_RT_VISIBLE_DEVICES."
        )
    return "No visible CUDA GPUs. Check nvidia-smi and CUDA_VISIBLE_DEVICES."

def _resolve_dtype(dtype_spec):
    """字符串或 torch.dtype → torch.dtype。支持 bf16/fp16/fp32 等。"""
    import torch

    if isinstance(dtype_spec, torch.dtype):
        return dtype_spec
    if isinstance(dtype_spec, str):
        key = dtype_spec.strip().lower()
        table = {
            "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
            "fp16": torch.float16, "float16": torch.float16, "f16": torch.float16, "half": torch.float16,
            "float32": torch.float32, "fp32": torch.float32, "f32": torch.float32, "float": torch.float32,
        }
        return table.get(key, torch.float32)
    return torch.float32


def _visible_physical_gpu_ids() -> list[int]:
    """兼容旧调用：返回当前后端可见设备列表。"""
    return visible_accelerator_device_ids()
