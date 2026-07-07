"""模型推理参数：model_name、采样、dtype、量化等。"""

from __future__ import annotations

from typing import Any, Optional


class ModelConfig:
    def __init__(
        self,
        model_name: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 256,
        max_batch_size: int = 8,
        dtype: str = "bfloat16",
        use_chat_template: bool = True,
        quantization: Optional[str] = None,
        lora_config: Any = None,
        max_model_len: Optional[int] = None,
        min_tokens: int = 0,
        gpu_memory_utilization: Optional[float] = None,
        enable_prefix_caching: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.common_message = ""
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.max_batch_size = max_batch_size
        self.dtype = dtype
        self.quantization = quantization
        self.lora_config = lora_config
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.use_chat_template = use_chat_template
        # vLLM option.  When enabled and prompts share prefixes, local placement
        # decisions from SAI-LP can translate into real prefix/KV-cache reuse.
        self.enable_prefix_caching = enable_prefix_caching
