"""测试用 Worker：不依赖 vLLM/真实加速卡，echo 输入。MFE_TEST_WORKER_DELAY 可设模拟延迟（秒）。"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List

from mfe.components import ExecuteInfo
from mfe.config import is_verbose

logger = logging.getLogger(__name__)


class TestWorker:
    def __init__(self, id: int, physical_device_id: int, cmd_queue: "Any", result_queue: "Any") -> None:
        self.id = id
        self.physical_device_id = physical_device_id
        self.cmd_queue = cmd_queue
        self.response_queue = result_queue
        logger.info("TestWorker[%s] initialized (echo mode)", id)

    def execute(self, exe_info: ExecuteInfo) -> Dict[str, Any]:
        op = exe_info.op
        op_name = getattr(op, "id", "op_unknown")
        is_duplicate = getattr(op, "is_duplicate", False)
        delay = float(os.environ.get("MFE_TEST_WORKER_DELAY", "20"))
        if delay > 0:
            time.sleep(delay)
        start = time.perf_counter()
        results: List[Dict[str, Any]] = []
        for qid, prompt in zip(exe_info.query_ids, exe_info.prompts):
            text = prompt if isinstance(prompt, str) else str(prompt)
            results.append({"id": qid, "output": text, "benchmark": (start, time.perf_counter())})
        elapsed = time.perf_counter() - start
        benchmark = {"init_time": 0.0, "prefill_time": 0.0, "generate_time": 0.0} if is_duplicate else {"init_time": 0.0, "prefill_time": 0.0, "generate_time": elapsed}
        return {"item": results, "op_name": op_name, "benchmark": benchmark}

    def exit(self) -> str:
        logger.info("TestWorker[%s] exited.", self.id)
        return "TestWorker exited."

    def run(self, debug: bool = False) -> None:
        while True:
            msg = self.cmd_queue.get()
            if isinstance(msg, tuple):
                command, params = msg
            elif isinstance(msg, dict):
                command = msg.get("command")
                params = msg.get("params", ())
            else:
                self.response_queue.put({"command": "error", "result": "Unsupported message format", "elapsed_time": 0.0})
                continue
            if command == "exit":
                self.response_queue.put({"command": "exit", "result": self.exit(), "elapsed_time": 0.0})
                break
            if is_verbose() and command == "execute" and params:
                exe = params[0]
                op_id = getattr(exe.op, "id", "?")
                qids = getattr(exe, "query_ids", [])
                prompts = getattr(exe, "prompts", [])
                p0 = (prompts[0][:50] + "...") if prompts and len(prompts[0]) > 50 else (prompts[0] if prompts else "")
                print(f"[Worker {self.id}] recv execute op={op_id} query_ids={qids} n_prompts={len(prompts)} prompt0={p0!r}", flush=True)
            func = getattr(self, command, None)
            if not callable(func):
                self.response_queue.put({"command": "error", "result": f"Unknown command: {command}", "elapsed_time": 0.0})
                continue
            start = time.perf_counter()
            try:
                result = func(*params) if params else func()
                elapsed = time.perf_counter() - start
                self.response_queue.put({"command": command, "result": result, "elapsed_time": elapsed})
                if is_verbose() and command == "execute" and isinstance(result, dict):
                    print(f"[Worker {self.id}] sent result op_name={result.get('op_name', '?')} elapsed={elapsed:.3f}s", flush=True)
            except Exception as e:
                if debug:
                    raise
                op_id = "?"
                if params:
                    exe = params[0]
                    op_id = getattr(getattr(exe, "op", None), "id", "?")
                self.response_queue.put({
                    "command": "error",
                    "result": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "op_name": op_id,
                        "worker_id": self.id,
                    },
                    "elapsed_time": 0.0,
                })
