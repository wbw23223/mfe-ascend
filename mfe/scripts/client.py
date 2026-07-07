#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多请求 Client：submit/status，从 parquet 数据集读取并测试。"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
import multiprocessing as mp
from typing import Any, Dict, List, Optional

_script_dir = os.path.dirname(os.path.abspath(__file__))
_package_root = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_package_root)
sys.path.insert(0, _project_root)

from mfe.serve import run_server
from mfe.config import is_verbose, set_verbose
from mfe.runtime import RuntimeConfig, collect_run_info
from mfe.scripts.process_datasets import PROCESSORS

DATASET_NAMES = ("drop", "gsm8k", "hotpotqa", "math")


def load_questions_from_parquet(
    dataset_name: str,
    data_dir: str,
    yaml_name: str = "adv_reason_3.yaml",
    n: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """从 data/{dataset}/{dataset}.parquet 读取前 n 个问题。"""
    dataset_name = dataset_name.lower()
    if dataset_name not in PROCESSORS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(PROCESSORS.keys())}")
    rows = PROCESSORS[dataset_name](data_dir, n)
    if not yaml_name.endswith(".yaml"):
        yaml_name = f"{yaml_name}.yaml"
    for r in rows:
        r["yaml"] = yaml_name
    return rows


def _to_json_safe(obj: Any) -> Any:
    """将 numpy 等类型转为 JSON 可序列化格式，避免 json.dump 报错。"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(x) for x in obj]
    return obj


def _json_default(obj: Any) -> Any:
    """处理 numpy 等不可 JSON 序列化的类型。"""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _zero_timestamps(results: List[Dict[str, Any]]) -> None:
    """将所有 benchmark、submit/arrive/start/done_time 减去最小值，使时间从 0 开始。"""
    all_ts: List[float] = []
    for r in results:
        sbt = r.get("submit_time")
        if sbt is not None:
            all_ts.append(float(sbt))
        at = r.get("arrive_time")
        if at is not None:
            all_ts.append(float(at))
        st = r.get("start_time")
        if st is not None:
            all_ts.append(float(st))
        dt = r.get("done_time")
        if dt is not None:
            all_ts.append(float(dt))
        for vals in (r.get("benchmark") or {}).values():
            if isinstance(vals, (list, tuple)) and len(vals) >= 2:
                all_ts.extend([float(vals[0]), float(vals[1])])
    if not all_ts:
        return
    min_ts = min(all_ts)
    for r in results:
        if r.get("submit_time") is not None:
            r["submit_time"] = float(r["submit_time"]) - min_ts
        if r.get("arrive_time") is not None:
            r["arrive_time"] = float(r["arrive_time"]) - min_ts
        if r.get("start_time") is not None:
            r["start_time"] = float(r["start_time"]) - min_ts
        if r.get("done_time") is not None:
            r["done_time"] = float(r["done_time"]) - min_ts
        bench = r.get("benchmark") or {}
        r["benchmark"] = {k: [float(v[0]) - min_ts, float(v[1]) - min_ts] for k, v in bench.items()}


class Client:
    def __init__(
        self,
        request_queue: mp.Queue,
        response_queue: mp.Queue,
    ) -> None:
        self._req_q = request_queue
        self._resp_q = response_queue
        self._pending: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._consumer.start()

    def _consume(self) -> None:
        while True:
            try:
                msg = self._resp_q.get()
            except Exception:
                break
            if not isinstance(msg, dict):
                continue
            req_id = msg.get("req_id", "")
            with self._lock:
                if req_id in self._pending:
                    self._pending[req_id]["result"] = msg
                    self._pending[req_id]["event"].set()

    def submit(self, dag: str, input_text: str) -> str:
        req_id = str(uuid.uuid4())
        if is_verbose():
            prompt_preview = (input_text or "")[:50] + ("..." if len(input_text or "") > 50 else "")
            print(f"[REQ] id={req_id[:8]} template={dag} prompt={prompt_preview!r}", flush=True)
        ev = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": ev, "result": None}
        self._req_q.put({"req_id": req_id, "command": "submit", "dag": dag, "input": input_text})
        ev.wait(timeout=10.0)
        with self._lock:
            r = self._pending.pop(req_id, {}).get("result")
        if r and r.get("error") is None:
            return r.get("result", {}).get("uid", "")
        raise RuntimeError(r.get("error", "submit failed") if r else "timeout")

    def status(self, uid: str) -> Optional[Dict[str, Any]]:
        req_id = str(uuid.uuid4())
        ev = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": ev, "result": None}
        self._req_q.put({"req_id": req_id, "command": "status", "uid": uid})
        ev.wait(timeout=5.0)
        with self._lock:
            r = self._pending.pop(req_id, {}).get("result")
        if r and r.get("error") is None:
            return r.get("result")
        return None

    def close(self) -> None:
        self._req_q.put({"command": "exit"})


def _strip_chat_template(raw: str) -> str:
    """从 Llama chat template 原始输出中提取最后一个 assistant 回复。"""
    if not raw:
        return ""
    marker = "assistant\n\n"
    idx = raw.rfind(marker)
    if idx < 0:
        return raw.strip()
    result = raw[idx + len(marker):]
    if result.endswith("<|eot_id|>"):
        result = result[:-10]
    return result.strip()


def _extract_final_answer(st: Dict[str, Any]) -> str:
    """从 status 的 op_output 中取最后节点新增的输出，并剔除 chat template，得到纯文本答案。"""
    op_output = st.get("op_output", {}) or {}
    benchmark = st.get("benchmark", {}) or {}
    if not op_output:
        return ""
    if not benchmark:
        return _strip_chat_template(next(iter(op_output.values()), ""))
    sorted_ops = sorted(benchmark.keys(), key=lambda k: benchmark[k][1] if k in benchmark else 0)
    last_op = sorted_ops[-1]
    full_last = op_output.get(last_op, "")
    if len(sorted_ops) > 1:
        prev_op = sorted_ops[-2]
        prefix = op_output.get(prev_op, "")
        if full_last.startswith(prefix):
            full_last = full_last[len(prefix):].strip()
    return _strip_chat_template(full_last)


def _build_error_result(item: Dict[str, Any], uid: str, st: Dict[str, Any], submit_time: float) -> Dict[str, Any]:
    q_full = item.get("question", "")
    out_item: Dict[str, Any] = {
        "question_preview": q_full[:100] if isinstance(q_full, str) else str(q_full)[:100],
        "question": q_full,
        "yaml": item.get("yaml", ""),
        "mfe_answer": "",
        "benchmark": st.get("benchmark") or {},
        "op_durations": {},
        "run_time": 0.0,
        "end_op_name": None,
        "worker_assignments": st.get("worker_assignments") or {},
        "submit_time": submit_time,
        "total_answer_time": st.get("total_answer_time"),
        "arrive_time": st.get("arrive_time"),
        "start_time": None,
        "service_time": None,
        "idle_time": None,
        "done_time": st.get("done_time"),
        "latency": None,
        "uid": uid,
        "status": "error",
        "error_type": st.get("error_type"),
        "error_message": st.get("error_message"),
        "failed_op": st.get("failed_op"),
        "worker_id": st.get("worker_id"),
    }
    for k, v in item.items():
        if k not in out_item:
            out_item[k] = v
    return out_item


def run_data_test(
    client: Client,
    questions: List[Dict[str, Any]],
    send_interval: float = 0.0,
) -> List[Dict[str, Any]]:
    """发送所有问题、等待完成、返回结果列表。questions 每项需有 question、yaml。"""
    uids: List[str] = []
    submit_times: Dict[str, float] = {}
    for i, item in enumerate(questions):
        if i > 0 and send_interval > 0:
            time.sleep(send_interval)
        q = item.get("question", "")
        yaml_name = item.get("yaml", "adv_reason_3.yaml")
        if not yaml_name.endswith(".yaml"):
            yaml_name = f"{yaml_name}.yaml"
        submit_time = time.perf_counter()
        uid = client.submit(yaml_name, q)
        uids.append(uid)
        submit_times[uid] = submit_time

    completed: Dict[str, Dict[str, Any]] = {}
    last_progress = 0.0
    while len(completed) < len(uids):
        for i, uid in enumerate(uids):
            if uid in completed:
                continue
            st = client.status(uid)
            if st and st.get("status") == "error":
                completed[uid] = _build_error_result(questions[i], uid, st, submit_times.get(uid, time.perf_counter()))
                msg = st.get("error_message") or "unknown error"
                print(f"  [{i+1}/{len(uids)}] uid={uid[:8]}... error: {msg}", flush=True)
            elif st and st.get("status") == "completed":
                item = questions[i]
                mfe_answer = _extract_final_answer(st)
                arrive_time = st.get("arrive_time")
                done_time = st.get("done_time")
                bench = st.get("benchmark") or {}
                start_time = min(float(v[0]) for v in bench.values()) if bench else None
                idle_time = (float(start_time) - float(arrive_time)) if (arrive_time is not None and start_time is not None) else None
                latency = (float(done_time) - float(arrive_time)) if (arrive_time is not None and done_time is not None) else None
                op_durations = {
                    op_name: (float(v[1]) - float(v[0]))
                    for op_name, v in bench.items()
                    if isinstance(v, (list, tuple)) and len(v) >= 2
                }
                run_time = sum(op_durations.values())
                end_op_name = max(bench.keys(), key=lambda k: bench[k][1]) if bench else None
                service_time = (
                    (float(done_time) - float(start_time))
                    if (done_time is not None and start_time is not None)
                    else None
                )
                worker_assignments = st.get("worker_assignments") or {}
                q_full = item.get("question", "")
                # 题目信息放最前：保留 preview 便于浏览，同时保留完整 question 便于追溯
                out_item: Dict[str, Any] = {
                    "question_preview": q_full[:100] if isinstance(q_full, str) else str(q_full)[:100],
                    "question": q_full,
                    "yaml": item.get("yaml", ""),
                }
                for k, v in item.items():
                    if k not in out_item:
                        out_item[k] = v
                out_item["mfe_answer"] = mfe_answer
                out_item["benchmark"] = bench
                out_item["op_durations"] = op_durations
                out_item["run_time"] = run_time
                out_item["end_op_name"] = end_op_name
                out_item["worker_assignments"] = worker_assignments
                out_item["submit_time"] = submit_times.get(uid)
                out_item["total_answer_time"] = st.get("total_answer_time")
                out_item["arrive_time"] = arrive_time
                out_item["start_time"] = start_time
                out_item["service_time"] = service_time
                out_item["idle_time"] = idle_time
                out_item["done_time"] = done_time
                out_item["latency"] = latency
                out_item["uid"] = uid
                out_item["status"] = "completed"
                completed[uid] = out_item
                if is_verbose() and st.get("total_answer_time") is not None:
                    print(f"  [{i+1}/{len(uids)}] uid={uid[:8]}... completed in {st['total_answer_time']:.2f}s", flush=True)
        now = time.perf_counter()
        if len(completed) < len(uids) and now - last_progress >= 10.0:
            print(f"  ... waiting: {len(completed)}/{len(uids)} completed", flush=True)
            last_progress = now
        time.sleep(0.5)

    return [completed[uid] for uid in uids]


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="MFE 多请求 Client")
    p.add_argument("--dataset", required=True, type=str.lower, choices=DATASET_NAMES, help="数据集：drop, gsm8k, hotpotqa, math")
    p.add_argument("-n", "--num", type=int, default=None, help="使用前 n 个问题测试，不指定则用全部。保存为 {dataset}_{yaml}_result_{n}.json")
    p.add_argument("--templates-dir", default="templates", help="工作流 YAML 目录")
    p.add_argument("--yaml", default="adv_reason_3.yaml", help="YAML 模板，如 adv_reason_4m.yaml。可指定不同 yaml 跑同一数据集，结果文件名会带上 yaml 名")
    p.add_argument("--model-path", default=None, help="覆盖模板中所有 op 的 model 字段；也可用 MFE_MODEL_PATH")
    p.add_argument("--data-dir", default=None, help="数据目录；默认 MFE_DATA_DIR 或项目根目录 data")
    p.add_argument("--output-dir", default=None, help="结果输出目录；默认 MFE_OUTPUT_DIR 或 data/{dataset}")
    p.add_argument("--device-ids", default=None, help="可见设备 ID，如 0 或 0,1；也可用 MFE_DEVICE_IDS")
    p.add_argument("--accelerator", default=None, choices=("ascend", "cuda"), help="推理后端；默认 MFE_ACCELERATOR 或 ascend")
    p.add_argument("--max-model-len", type=int, default=None, help="覆盖 vLLM max_model_len；也可用 MFE_MAX_MODEL_LEN")
    p.add_argument("--offline", action="store_true", help="启用 HuggingFace/Transformers/Datasets 离线模式")
    p.add_argument("--send-interval", type=float, default=0.0, help="发送间隔（秒）")
    p.add_argument("--worker-delay", type=float, default=None, help="TestWorker 模拟延迟（秒）")
    p.add_argument("--test-worker", action="store_true", help="使用 TestWorker")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    if args.verbose:
        set_verbose(True)
        os.environ["MFE_VERBOSE"] = "1"
    if args.worker_delay is not None:
        os.environ["MFE_TEST_WORKER_DELAY"] = str(args.worker_delay)
    if args.max_model_len is not None:
        os.environ["MFE_MAX_MODEL_LEN"] = str(args.max_model_len)

    req_q = mp.Queue()
    resp_q = mp.Queue()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(root)
    runtime = RuntimeConfig.from_values(
        accelerator=args.accelerator,
        device_ids=args.device_ids,
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        offline=args.offline or None,
        project_root=root,
    )
    runtime.apply()
    templates_abs = os.path.abspath(args.templates_dir)

    proc = mp.Process(
        target=run_server,
        args=(req_q, resp_q, templates_abs, args.test_worker),
        daemon=False,
    )
    proc.start()

    client = Client(req_q, resp_q)
    try:
        data_dir = runtime.data_dir
        questions = load_questions_from_parquet(args.dataset, data_dir, args.yaml, args.num)
        if not questions:
            expected_path = os.path.join(data_dir, args.dataset, f"{args.dataset}.parquet")
            print(f"No data for dataset {args.dataset}. Expected parquet file: {expected_path}")
            return
        print(f"Loaded {len(questions)} questions from data/{args.dataset}/{args.dataset}.parquet")
        results = run_data_test(client, questions, send_interval=args.send_interval)
        _zero_timestamps(results)
        run_info = collect_run_info(runtime, cwd=root)
        for item in results:
            item["run_info"] = run_info
        results = _to_json_safe(results)  # 转换 numpy 等类型，避免 json.dump 报错
        out_dir = runtime.output_dir or os.path.join(data_dir, args.dataset)
        os.makedirs(out_dir, exist_ok=True)
        yaml_base = args.yaml.replace(".yaml", "") if args.yaml else "default"
        if args.num is not None:
            out_name = f"{args.dataset}_{yaml_base}_result_{args.num}.json"
        else:
            out_name = f"{args.dataset}_{yaml_base}_result.json"
        out_path = os.path.join(out_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=_json_default)
        print(f"Saved {len(results)} results to {out_path}")
        if any(r.get("status") == "error" for r in results):
            raise SystemExit(2)
    finally:
        client.close()
        proc.join(timeout=5.0)


if __name__ == "__main__":
    main()
