"""DAG 节点：Operator、Benchmark。"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class Benchmark:
    def __init__(self) -> None:
        self.init_time = 0.0
        self.prefill_time = 0.0
        self.generate_time = 0.0

    def total_time(self) -> float:
        return self.init_time + self.prefill_time + self.generate_time

    def update(self, d: Dict[str, float]) -> None:
        self.init_time += d.get("init_time", 0.0)
        self.prefill_time += d.get("prefill_time", 0.0)
        self.generate_time += d.get("generate_time", 0.0)

    def __str__(self) -> str:
        return (
            f"Init: {self.init_time}, Prefill: {self.prefill_time}, "
            f"Generate: {self.generate_time}, Total: {self.total_time()}"
        )


class Operator:
    def __init__(
        self,
        id: Optional[str] = None,
        prompt: Optional[str] = None,
        model_config: Any = None,
        keep_cache: bool = False,
    ) -> None:
        self.id = id if id is not None else str(uuid.uuid4())
        self.input_ops: List["Operator"] = []
        self.output_ops: List["Operator"] = []
        self.prompt = prompt
        self.model_config = model_config
        self.benchmark = Benchmark()
        self.is_duplicate = False
        self.keep_cache = keep_cache

        # Optional SAIL/SAI-LP scheduling metadata.  Existing YAML files do not
        # need these fields; parser.py fills them from op specs when present.
        self.max_distance = -1
        self.reuse_from: List[Any] = []
        self.reuse_group: Optional[str] = None
        self.eligible_devices: Optional[List[int]] = None
        self.sailp: Dict[str, Any] = {}
