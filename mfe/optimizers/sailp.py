"""SAIL/SAI-LP workflow scheduler for MFE Ascend.

This module implements an admission-time, state-aware DAG scheduler inspired by
"SAIL: State-Affinity Insertion with LP Guidance for Multi-Agent LLM Workflow
Scheduling".  The implementation intentionally lives above the vLLM worker:
it plans where every Operator should run and which upstream Operator it should
try to keep state-affine with.  The current vLLM worker still owns the concrete
KV/prefix-cache mechanism; this scheduler biases placement/order so automatic
prefix caching can be effective when enabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
import math
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from mfe.components import Operator

logger = getLogger(__name__)

_EPS = 1e-9
ReuseKey = Tuple[str, str]
DataKey = Tuple[str, str, int, int]
BenefitKey = Tuple[str, str, int]
StateDelayKey = Tuple[str, str, int, int]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid float env %s=%r", name, raw)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ignoring invalid integer env %s=%r", name, raw)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_executor_list(value: Any, all_executors: Sequence[int]) -> Optional[List[int]]:
    """Parse YAML/env executor lists such as [0, 1], "0,1", or "all"."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw or raw == "all":
            return list(all_executors)
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        parsed = [int(p) for p in parts]
    elif isinstance(value, int):
        parsed = [value]
    else:
        parsed = [int(x) for x in value]
    allowed = set(all_executors)
    return [x for x in parsed if x in allowed]


def _lookup(mapping: Mapping[Any, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


@dataclass
class ScheduleStep:
    op_id: str
    worker_id: int
    planned_start: float
    planned_end: float
    estimated_duration: float
    cold_duration: float
    reuse_from: Optional[str] = None
    reuse_mode: str = "cold"
    priority: float = 0.0
    order_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op_id": self.op_id,
            "worker_id": self.worker_id,
            "planned_start": self.planned_start,
            "planned_end": self.planned_end,
            "estimated_duration": self.estimated_duration,
            "cold_duration": self.cold_duration,
            "reuse_from": self.reuse_from,
            "reuse_mode": self.reuse_mode,
            "priority": self.priority,
            "order_index": self.order_index,
        }


@dataclass
class SchedulePlan:
    steps: Dict[str, ScheduleStep]
    timelines: Dict[int, List[str]]
    makespan: float
    guidance_method: str
    warnings: List[str] = field(default_factory=list)

    def worker_for(self, op_id: str) -> Optional[int]:
        step = self.steps.get(op_id)
        return None if step is None else step.worker_id

    def priority_for(self, op_id: str) -> Tuple[float, float, int]:
        step = self.steps.get(op_id)
        if step is None:
            return (math.inf, math.inf, 10**9)
        return (step.planned_start, step.planned_end, step.order_index)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "makespan": self.makespan,
            "guidance_method": self.guidance_method,
            "timelines": {str(k): list(v) for k, v in self.timelines.items()},
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "warnings": list(self.warnings),
        }


@dataclass
class _Instance:
    ops: Dict[str, Operator]
    tasks: List[str]
    topo: List[str]
    data_edges: List[Tuple[str, str]]
    preds: Dict[str, Set[str]]
    succs: Dict[str, Set[str]]
    executors: List[int]
    eligible: Dict[str, List[int]]
    p_cold: Dict[Tuple[str, int], float]
    data_delay: Dict[DataKey, float]
    reuse_edges: List[ReuseKey]
    reuse_by_target: Dict[str, List[str]]
    b: Dict[BenefitKey, float]
    h: Dict[StateDelayKey, float]
    p_bar: float
    upper_bound: float


@dataclass
class _Guidance:
    method: str
    p_hat: Dict[str, float]
    d_hat: Dict[Tuple[str, str], float]
    L: Dict[str, float]
    tail: Dict[str, float]
    alpha: Dict[ReuseKey, float]
    a_out: Dict[str, float]
    warnings: List[str] = field(default_factory=list)


@dataclass
class _Skeleton:
    mu: Dict[str, int]
    rho: Dict[str, Optional[str]]
    pi: Dict[int, List[str]]

    def copy(self) -> "_Skeleton":
        return _Skeleton(
            mu=dict(self.mu),
            rho=dict(self.rho),
            pi={m: list(seq) for m, seq in self.pi.items()},
        )


@dataclass
class _DecodeResult:
    feasible: bool
    start: Dict[str, float] = field(default_factory=dict)
    finish: Dict[str, float] = field(default_factory=dict)
    duration: Dict[str, float] = field(default_factory=dict)
    makespan: float = math.inf
    topo: List[str] = field(default_factory=list)
    reason: str = ""


class SAILPScheduler:
    """LP-guided State-Affinity Insertion scheduler.

    The scheduler accepts the existing MFE `Operator` DAG and returns a
    `SchedulePlan` that maps every op to a worker/device.  It supports the
    paper's three stages:

    1. a reduced placement-reuse LP when SciPy is available;
    2. state-affinity insertion over executor timelines;
    3. bounded makespan-executor refinement.

    If SciPy is not installed or the LP is disabled, the planner falls back to
    HEFT-style ranks plus closed-form reuse affinity, matching the paper's
    SAI-NoLP ablation rather than failing at runtime.
    """

    def __init__(
        self,
        num_executors: int,
        *,
        executor_ids: Optional[Sequence[int]] = None,
        refinement_rounds: Optional[int] = None,
    ) -> None:
        if executor_ids is None:
            executor_ids = list(range(num_executors))
        self.executors = [int(x) for x in executor_ids]
        if not self.executors:
            raise ValueError("SAILPScheduler requires at least one executor")
        self.refinement_rounds = (
            _env_int("MFE_SAILP_REFINEMENT_ROUNDS", 2)
            if refinement_rounds is None
            else int(refinement_rounds)
        )
        # Default scoring weights from the SAI-LP paper, exposed as env knobs.
        self.w_mk = _env_float("MFE_SAILP_W_MK", 1.0)
        self.w_dn = _env_float("MFE_SAILP_W_DN", 0.3)
        self.w_omega = _env_float("MFE_SAILP_W_OMEGA", 1.5)
        self.w_pr = _env_float("MFE_SAILP_W_PR", 0.3)
        self.w_lc = _env_float("MFE_SAILP_W_LC", 0.15)
        self.w_rs = _env_float("MFE_SAILP_W_RS", 0.3)
        self.beta = _env_float("MFE_SAILP_BETA", 0.20)
        self.tau = _env_float("MFE_SAILP_TAU", 0.25)

    def plan(self, ops: Dict[str, Operator]) -> SchedulePlan:
        inst = self._build_instance(ops)
        guidance = self._solve_guidance(inst)
        skeleton = self._construct(inst, guidance)
        if self.refinement_rounds > 0:
            skeleton = self._refine(inst, guidance, skeleton, self.refinement_rounds)
        decoded = self._decode(inst, skeleton)
        if not decoded.feasible or decoded.makespan > inst.upper_bound + _EPS:
            warnings = list(guidance.warnings)
            warnings.append(
                "SAI-LP produced an invalid or too-large skeleton; used cold EFT fallback"
            )
            skeleton = self._cold_eft_fallback(inst)
            decoded = self._decode(inst, skeleton)
            guidance_method = f"{guidance.method}+fallback"
        else:
            warnings = list(guidance.warnings)
            guidance_method = guidance.method
        return self._to_plan(inst, guidance, skeleton, decoded, guidance_method, warnings)

    # ------------------------------------------------------------------
    # Instance construction
    # ------------------------------------------------------------------
    def _build_instance(self, ops: Dict[str, Operator]) -> _Instance:
        tasks = list(ops.keys())
        preds: Dict[str, Set[str]] = {u: set() for u in tasks}
        succs: Dict[str, Set[str]] = {u: set() for u in tasks}
        for u, op in ops.items():
            for parent in getattr(op, "input_ops", []) or []:
                pid = str(parent.id)
                if pid in ops:
                    preds[u].add(pid)
                    succs[pid].add(u)
            for child in getattr(op, "output_ops", []) or []:
                cid = str(child.id)
                if cid in ops:
                    preds[cid].add(u)
                    succs[u].add(cid)
        data_edges = sorted((u, v) for u in tasks for v in succs[u])
        topo = self._topological_order(tasks, preds, succs)
        descendants = self._descendants(tasks, succs)

        eligible: Dict[str, List[int]] = {}
        p_cold: Dict[Tuple[str, int], float] = {}
        speeds = self._device_speeds()
        for u in tasks:
            op = ops[u]
            spec = self._sailp_spec(op)
            raw_eligible = _lookup(
                spec,
                "eligible_devices",
                "eligible_executors",
                "devices",
                "executors",
                default=getattr(op, "eligible_devices", None),
            )
            parsed = _parse_executor_list(raw_eligible, self.executors)
            if not parsed:
                parsed = list(self.executors)
            eligible[u] = parsed

            device_times = _lookup(spec, "device_time", "device_times", default=None)
            base = self._estimate_cold_time(op, spec)
            for m in parsed:
                if isinstance(device_times, Mapping):
                    val = _lookup(device_times, str(m), m, default=None)
                    tm = _as_float(val, None)
                    p_cold[(u, m)] = max(_EPS, tm if tm is not None else base / speeds.get(m, 1.0))
                else:
                    p_cold[(u, m)] = max(_EPS, base / speeds.get(m, 1.0))

        data_delay: Dict[DataKey, float] = {}
        for u, v in data_edges:
            delay = self._edge_data_delay(ops[u], ops[v])
            for m in eligible[u]:
                for n in eligible[v]:
                    data_delay[(u, v, m, n)] = 0.0 if m == n else delay

        reuse_edges, reuse_specs = self._collect_reuse_edges(ops, topo, descendants)
        reuse_by_target: Dict[str, List[str]] = {u: [] for u in tasks}
        for r, u in reuse_edges:
            reuse_by_target[u].append(r)
        for u in reuse_by_target:
            reuse_by_target[u].sort(key=lambda r: topo.index(r) if r in topo else len(topo))

        b: Dict[BenefitKey, float] = {}
        h: Dict[StateDelayKey, float] = {}
        for r, u in reuse_edges:
            rspec = reuse_specs.get((r, u), {})
            for m in eligible[u]:
                benefit = self._reuse_benefit(ops[r], ops[u], rspec, p_cold[(u, m)], m)
                b[(r, u, m)] = min(max(0.0, benefit), p_cold[(u, m)] - _EPS)
            remote_delay = self._state_access_delay(ops[r], ops[u], rspec)
            for n in eligible[r]:
                for m in eligible[u]:
                    h[(r, u, n, m)] = 0.0 if n == m else remote_delay

        p_bar = sum(p_cold.values()) / max(1, len(p_cold))
        max_d = max(data_delay.values()) if data_delay else 0.0
        # A loose no-reuse upper bound.  It is only used for LP big-M and for
        # sanity-checking; the fallback below is the executable schedule.
        upper_bound = sum(max(p_cold[(u, m)] for m in eligible[u]) for u in tasks) + len(tasks) * max_d
        return _Instance(
            ops=ops,
            tasks=tasks,
            topo=topo,
            data_edges=data_edges,
            preds=preds,
            succs=succs,
            executors=list(self.executors),
            eligible=eligible,
            p_cold=p_cold,
            data_delay=data_delay,
            reuse_edges=reuse_edges,
            reuse_by_target=reuse_by_target,
            b=b,
            h=h,
            p_bar=max(_EPS, p_bar),
            upper_bound=max(_EPS, upper_bound),
        )

    def _sailp_spec(self, op: Operator) -> Dict[str, Any]:
        spec: Dict[str, Any] = {}
        for attr in ("sailp", "profile", "scheduler"):
            val = getattr(op, attr, None)
            if isinstance(val, Mapping):
                spec.update(dict(val))
        return spec

    def _estimate_cold_time(self, op: Operator, spec: Mapping[str, Any]) -> float:
        val = _lookup(spec, "cold_time", "estimated_time", "duration", "cost", default=None)
        parsed = _as_float(val, None)
        if parsed is not None:
            return max(_EPS, parsed)
        cfg = getattr(op, "model_config", None)
        max_tokens = _as_float(getattr(cfg, "max_tokens", None), 0.0) or 0.0
        token_time = _env_float("MFE_SAILP_TOKEN_TIME", 0.001)
        default = _env_float("MFE_SAILP_DEFAULT_OP_TIME", 1.0)
        return max(_EPS, default + max_tokens * token_time)

    def _edge_data_delay(self, src: Operator, dst: Operator) -> float:
        src_spec = self._sailp_spec(src)
        dst_spec = self._sailp_spec(dst)
        val = _lookup(dst_spec, "input_data_delay", "data_delay", default=None)
        if val is None:
            val = _lookup(src_spec, "output_data_delay", "data_delay", default=None)
        parsed = _as_float(val, None)
        if parsed is not None:
            return max(0.0, parsed)
        return max(0.0, _env_float("MFE_SAILP_CROSS_DATA_DELAY", 0.05))

    def _collect_reuse_edges(
        self,
        ops: Dict[str, Operator],
        topo: Sequence[str],
        descendants: Dict[str, Set[str]],
    ) -> Tuple[List[ReuseKey], Dict[ReuseKey, Dict[str, Any]]]:
        edges: Set[ReuseKey] = set()
        specs: Dict[ReuseKey, Dict[str, Any]] = {}
        # Explicit target-side reuse_from annotations.
        for u, op in ops.items():
            raw = getattr(op, "reuse_from", None)
            if raw is None:
                raw = self._sailp_spec(op).get("reuse_from")
            for item in _as_list(raw):
                rspec: Dict[str, Any] = {}
                if isinstance(item, Mapping):
                    r = _lookup(item, "op_id", "op", "source", "source_op", "id", default=None)
                    rspec = dict(item)
                else:
                    r = item
                if r is None:
                    continue
                r = str(r)
                if r not in ops or r == u:
                    continue
                key = (r, u)
                edges.add(key)
                specs[key] = rspec
        # Reuse groups: every pair in the same group becomes a candidate after
        # causal filtering.  This is convenient for shared-prefix workflows.
        groups: Dict[str, List[str]] = {}
        for u, op in ops.items():
            group = getattr(op, "reuse_group", None)
            if group is None:
                group = self._sailp_spec(op).get("reuse_group")
            if group is not None:
                groups.setdefault(str(group), []).append(u)
        topo_pos = {u: i for i, u in enumerate(topo)}
        for members in groups.values():
            members = sorted(set(members), key=lambda x: topo_pos.get(x, 10**9))
            for r in members:
                for u in members:
                    if r == u:
                        continue
                    # Prefer earlier topological candidates, but still rely on
                    # the causal filter below for correctness.
                    if topo_pos.get(r, 10**9) <= topo_pos.get(u, 10**9):
                        edges.add((r, u))
                        specs.setdefault((r, u), {})
        if _env_bool("MFE_SAILP_REUSE_INPUTS", False):
            for u, op in ops.items():
                for parent in getattr(op, "input_ops", []) or []:
                    r = str(parent.id)
                    if r in ops:
                        edges.add((r, u))
                        specs.setdefault((r, u), {})

        filtered: List[ReuseKey] = []
        for r, u in sorted(edges, key=lambda x: (topo_pos.get(x[1], 10**9), topo_pos.get(x[0], 10**9), x)):
            # Remove causally impossible candidates: if u reaches r in the data
            # graph, adding r -> u would close a directed cycle.
            if r in descendants.get(u, set()):
                continue
            filtered.append((r, u))
        return filtered, specs

    def _reuse_benefit(
        self,
        src: Operator,
        dst: Operator,
        rspec: Mapping[str, Any],
        cold: float,
        executor_id: int,
    ) -> float:
        device_benefit = _lookup(rspec, "device_benefit", "device_benefits", default=None)
        if isinstance(device_benefit, Mapping):
            val = _lookup(device_benefit, str(executor_id), executor_id, default=None)
            parsed = _as_float(val, None)
            if parsed is not None:
                return parsed
        val = _lookup(rspec, "benefit", "reuse_benefit", "saved_time", default=None)
        if val is None:
            dst_spec = self._sailp_spec(dst)
            val = _lookup(dst_spec, "reuse_benefit", "benefit", default=None)
        parsed = _as_float(val, None)
        if parsed is not None:
            return parsed
        warm = _as_float(_lookup(rspec, "warm_time", "warm_duration", default=None), None)
        if warm is not None:
            return max(0.0, cold - warm)
        shared_tokens = _as_float(_lookup(rspec, "shared_prefix_tokens", "prefix_tokens", default=None), None)
        if shared_tokens is not None:
            return shared_tokens * _env_float("MFE_SAILP_REUSE_TOKEN_TIME", _env_float("MFE_SAILP_TOKEN_TIME", 0.001))
        ratio = _env_float("MFE_SAILP_REUSE_BENEFIT_RATIO", 0.35)
        return cold * ratio

    def _state_access_delay(self, src: Operator, dst: Operator, rspec: Mapping[str, Any]) -> float:
        val = _lookup(rspec, "remote_delay", "state_delay", "state_access_delay", default=None)
        if val is None:
            dst_spec = self._sailp_spec(dst)
            val = _lookup(dst_spec, "remote_state_delay", "state_delay", default=None)
        parsed = _as_float(val, None)
        if parsed is not None:
            return max(0.0, parsed)
        state_size = _as_float(_lookup(rspec, "state_size", "kv_size", default=None), None)
        if state_size is not None:
            bw = _env_float("MFE_SAILP_REMOTE_STATE_BANDWIDTH", 1.0)
            if bw > _EPS:
                return max(0.0, state_size / bw)
        return max(0.0, _env_float("MFE_SAILP_REMOTE_STATE_DELAY", 0.10))

    def _device_speeds(self) -> Dict[int, float]:
        raw = os.environ.get("MFE_SAILP_DEVICE_SPEEDS", "").strip()
        if not raw:
            return {m: 1.0 for m in self.executors}
        speeds: Dict[int, float] = {}
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        for i, part in enumerate(parts):
            if ":" in part:
                dev, val = part.split(":", 1)
                speeds[int(dev.strip())] = max(_EPS, float(val.strip()))
            elif i < len(self.executors):
                speeds[self.executors[i]] = max(_EPS, float(part))
        for m in self.executors:
            speeds.setdefault(m, 1.0)
        return speeds

    # ------------------------------------------------------------------
    # LP guidance
    # ------------------------------------------------------------------
    def _solve_guidance(self, inst: _Instance) -> _Guidance:
        mode = os.environ.get("MFE_SAILP_USE_LP", "auto").strip().lower()
        if mode in {"0", "false", "no", "off", "heuristic", "none"}:
            return self._heuristic_guidance(inst, warning="LP guidance disabled by MFE_SAILP_USE_LP")
        try:
            return self._lp_guidance(inst)
        except Exception as exc:  # pragma: no cover - intentionally robust in production
            if mode in {"1", "true", "yes", "on", "required"}:
                logger.exception("SAI-LP guidance failed")
            return self._heuristic_guidance(inst, warning=f"LP guidance unavailable; used heuristic guidance ({type(exc).__name__}: {exc})")

    def _lp_guidance(self, inst: _Instance) -> _Guidance:
        import numpy as np
        from scipy.optimize import linprog

        z_idx: Dict[Tuple[str, int], int] = {}
        eta_idx: Dict[Tuple[str, str, int, int], int] = {}
        y_idx: Dict[ReuseKey, int] = {}
        s_idx: Dict[str, int] = {}
        c_idx: Dict[str, int] = {}
        idx = 0
        for u in inst.tasks:
            for m in inst.eligible[u]:
                z_idx[(u, m)] = idx
                idx += 1
        for r, u in inst.reuse_edges:
            for n in inst.eligible[r]:
                for m in inst.eligible[u]:
                    eta_idx[(r, u, n, m)] = idx
                    idx += 1
        for e in inst.reuse_edges:
            y_idx[e] = idx
            idx += 1
        for u in inst.tasks:
            s_idx[u] = idx
            idx += 1
        for u in inst.tasks:
            c_idx[u] = idx
            idx += 1
        t_idx = idx
        idx += 1
        nvar = idx

        h_max = max(inst.h.values()) if inst.h else 0.0
        d_max = max(inst.data_delay.values()) if inst.data_delay else 0.0
        H = inst.upper_bound + max(h_max, d_max, 1.0)
        lambda_r = _env_float("MFE_SAILP_LAMBDA_R", 0.05 / inst.p_bar)

        c = np.zeros(nvar, dtype=float)
        c[t_idx] = 1.0
        for (r, u), j in y_idx.items():
            min_h = min(inst.h[(r, u, n, m)] for n in inst.eligible[r] for m in inst.eligible[u])
            avg_net = sum(inst.b[(r, u, m)] - self.tau * min_h for m in inst.eligible[u]) / max(1, len(inst.eligible[u]))
            c[j] = -lambda_r * avg_net

        bounds: List[Tuple[float, Optional[float]]] = [(0.0, 1.0)] * (len(z_idx) + len(eta_idx) + len(y_idx))
        bounds += [(0.0, inst.upper_bound + H)] * (2 * len(inst.tasks) + 1)

        A_eq: List[List[float]] = []
        b_eq: List[float] = []
        A_ub: List[List[float]] = []
        b_ub: List[float] = []

        # Assignment: sum_m z_um = 1.
        for u in inst.tasks:
            row = np.zeros(nvar, dtype=float)
            for m in inst.eligible[u]:
                row[z_idx[(u, m)]] = 1.0
            A_eq.append(row.tolist())
            b_eq.append(1.0)

        # eta <= z_rn and eta <= z_um.
        for (r, u, n, m), j in eta_idx.items():
            row = np.zeros(nvar, dtype=float)
            row[j] = 1.0
            row[z_idx[(r, n)]] = -1.0
            A_ub.append(row.tolist())
            b_ub.append(0.0)
            row = np.zeros(nvar, dtype=float)
            row[j] = 1.0
            row[z_idx[(u, m)]] = -1.0
            A_ub.append(row.tolist())
            b_ub.append(0.0)

        # y_ru = sum_nm eta_r,u,n,m.
        for r, u in inst.reuse_edges:
            row = np.zeros(nvar, dtype=float)
            row[y_idx[(r, u)]] = 1.0
            for n in inst.eligible[r]:
                for m in inst.eligible[u]:
                    row[eta_idx[(r, u, n, m)]] -= 1.0
            A_eq.append(row.tolist())
            b_eq.append(0.0)

        # At most one reuse source per target.
        for u in inst.tasks:
            srcs = inst.reuse_by_target.get(u, [])
            if not srcs:
                continue
            row = np.zeros(nvar, dtype=float)
            for r in srcs:
                row[y_idx[(r, u)]] = 1.0
            A_ub.append(row.tolist())
            b_ub.append(1.0)

        # Completion: C_u = S_u + sum z p_cold - sum eta benefit.
        for u in inst.tasks:
            row = np.zeros(nvar, dtype=float)
            row[c_idx[u]] = 1.0
            row[s_idx[u]] = -1.0
            for m in inst.eligible[u]:
                row[z_idx[(u, m)]] -= inst.p_cold[(u, m)]
            for r in inst.reuse_by_target.get(u, []):
                for n in inst.eligible[r]:
                    for m in inst.eligible[u]:
                        row[eta_idx[(r, u, n, m)]] += inst.b[(r, u, m)]
            A_eq.append(row.tolist())
            b_eq.append(0.0)

        # Data timing: S_v >= C_u + d - H(2 - z_um - z_vn).
        for u, v in inst.data_edges:
            for m in inst.eligible[u]:
                for n in inst.eligible[v]:
                    row = np.zeros(nvar, dtype=float)
                    row[c_idx[u]] = 1.0
                    row[s_idx[v]] = -1.0
                    row[z_idx[(u, m)]] = H
                    row[z_idx[(v, n)]] = H
                    A_ub.append(row.tolist())
                    b_ub.append(2 * H - inst.data_delay[(u, v, m, n)])

        # State timing: S_u >= C_r + h - H(1 - eta).
        for (r, u, n, m), j in eta_idx.items():
            row = np.zeros(nvar, dtype=float)
            row[c_idx[r]] = 1.0
            row[s_idx[u]] = -1.0
            row[j] = H
            A_ub.append(row.tolist())
            b_ub.append(H - inst.h[(r, u, n, m)])

        # Makespan and upper-bound cuts.
        for u in inst.tasks:
            row = np.zeros(nvar, dtype=float)
            row[c_idx[u]] = 1.0
            row[t_idx] = -1.0
            A_ub.append(row.tolist())
            b_ub.append(0.0)
        row = np.zeros(nvar, dtype=float)
        row[t_idx] = 1.0
        A_ub.append(row.tolist())
        b_ub.append(inst.upper_bound)

        res = linprog(
            c,
            A_ub=np.asarray(A_ub, dtype=float) if A_ub else None,
            b_ub=np.asarray(b_ub, dtype=float) if b_ub else None,
            A_eq=np.asarray(A_eq, dtype=float) if A_eq else None,
            b_eq=np.asarray(b_eq, dtype=float) if b_eq else None,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(res.message)

        x = res.x
        z_val = {k: float(x[j]) for k, j in z_idx.items()}
        eta_val = {k: float(x[j]) for k, j in eta_idx.items()}
        p_hat: Dict[str, float] = {}
        for u in inst.tasks:
            val = sum(z_val[(u, m)] * inst.p_cold[(u, m)] for m in inst.eligible[u])
            for r in inst.reuse_by_target.get(u, []):
                for n in inst.eligible[r]:
                    for m in inst.eligible[u]:
                        val -= eta_val.get((r, u, n, m), 0.0) * inst.b[(r, u, m)]
            p_hat[u] = max(_EPS, val)

        d_hat: Dict[Tuple[str, str], float] = {}
        for u, v in inst.data_edges:
            d_hat[(u, v)] = sum(
                z_val[(u, m)] * z_val[(v, n)] * inst.data_delay[(u, v, m, n)]
                for m in inst.eligible[u]
                for n in inst.eligible[v]
            )

        alpha = {(r, u): float(x[y_idx[(r, u)]]) for r, u in inst.reuse_edges}
        return self._finish_guidance(inst, p_hat, d_hat, alpha, method="lp")

    def _heuristic_guidance(self, inst: _Instance, warning: Optional[str] = None) -> _Guidance:
        chosen: Dict[str, int] = {
            u: min(inst.eligible[u], key=lambda m: (inst.p_cold[(u, m)], m)) for u in inst.tasks
        }
        p_hat = {u: inst.p_cold[(u, chosen[u])] for u in inst.tasks}
        d_hat = {
            (u, v): inst.data_delay[(u, v, chosen[u], chosen[v])] for u, v in inst.data_edges
        }
        alpha: Dict[ReuseKey, float] = {}
        for u in inst.tasks:
            raw: Dict[str, float] = {}
            for r in inst.reuse_by_target.get(u, []):
                best_b = max(inst.b[(r, u, m)] for m in inst.eligible[u])
                best_h = min(inst.h[(r, u, n, m)] for n in inst.eligible[r] for m in inst.eligible[u])
                raw[r] = max(0.0, best_b - self.tau * best_h)
            denom = sum(raw.values())
            for r in inst.reuse_by_target.get(u, []):
                alpha[(r, u)] = raw[r] / denom if denom > _EPS else 0.0
        warnings = [warning] if warning else []
        return self._finish_guidance(inst, p_hat, d_hat, alpha, method="heuristic", warnings=warnings)

    def _finish_guidance(
        self,
        inst: _Instance,
        p_hat: Dict[str, float],
        d_hat: Dict[Tuple[str, str], float],
        alpha: Dict[ReuseKey, float],
        *,
        method: str,
        warnings: Optional[List[str]] = None,
    ) -> _Guidance:
        L: Dict[str, float] = {}
        for u in reversed(inst.topo):
            if inst.succs[u]:
                best_child = max((d_hat.get((u, v), 0.0) + L[v]) for v in inst.succs[u])
            else:
                best_child = 0.0
            L[u] = max(_EPS, p_hat[u] + best_child)
        tail = {u: max(0.0, L[u] - p_hat[u]) for u in inst.tasks}
        a_out: Dict[str, float] = {u: 0.0 for u in inst.tasks}
        for r, u in inst.reuse_edges:
            max_b = max(inst.b[(r, u, m)] for m in inst.eligible[u])
            a_out[r] += alpha.get((r, u), 0.0) * max_b
        return _Guidance(
            method=method,
            p_hat=p_hat,
            d_hat=d_hat,
            L=L,
            tail=tail,
            alpha=alpha,
            a_out=a_out,
            warnings=warnings or [],
        )

    # ------------------------------------------------------------------
    # Construction and refinement
    # ------------------------------------------------------------------
    def _construct(self, inst: _Instance, g: _Guidance) -> _Skeleton:
        skeleton = _Skeleton(mu={}, rho={}, pi={m: [] for m in inst.executors})
        start: Dict[str, float] = {}
        finish: Dict[str, float] = {}
        scheduled: Set[str] = set()
        topo_pos = {u: i for i, u in enumerate(inst.topo)}

        while len(scheduled) < len(inst.tasks):
            ready = [u for u in inst.topo if u not in scheduled and inst.preds[u] <= scheduled]
            if not ready:
                # Should not happen for a valid DAG, but keep a safe fallback.
                return self._cold_eft_fallback(inst)
            u = max(
                ready,
                key=lambda x: (g.L.get(x, 0.0) + self.beta * g.a_out.get(x, 0.0), -len(inst.eligible[x]), -topo_pos[x]),
            )
            best = None
            reuse_modes: List[Optional[str]] = [None] + [r for r in inst.reuse_by_target.get(u, []) if r in scheduled]
            for m in inst.eligible[u]:
                seq = skeleton.pi[m]
                gaps: List[Tuple[int, Optional[str], Optional[str]]] = []
                for pos in range(len(seq) + 1):
                    a = seq[pos - 1] if pos > 0 else None
                    s = seq[pos] if pos < len(seq) else None
                    gaps.append((pos, a, s))
                for rho in reuse_modes:
                    for pos, a, s in gaps:
                        cand = self._score_insertion(inst, g, skeleton, scheduled, start, finish, u, m, rho, pos, a, s)
                        if cand is None:
                            continue
                        if best is None or cand < best:
                            best = cand
            if best is None:
                # Conservative tail insertion, no reuse.
                best = self._fallback_insert_score(inst, g, skeleton, scheduled, start, finish, u)
            _, cand_finish, pos, m, rho, cand_start, duration = best
            skeleton.pi[m].insert(pos, u)
            skeleton.mu[u] = m
            skeleton.rho[u] = rho
            start[u] = cand_start
            finish[u] = cand_finish
            scheduled.add(u)
        return skeleton

    def _score_insertion(
        self,
        inst: _Instance,
        g: _Guidance,
        skeleton: _Skeleton,
        scheduled: Set[str],
        start: Dict[str, float],
        finish: Dict[str, float],
        u: str,
        m: int,
        rho: Optional[str],
        pos: int,
        a: Optional[str],
        s: Optional[str],
    ) -> Optional[Tuple[float, float, int, int, Optional[str], float, float]]:
        data_ready = 0.0
        for k in inst.preds[u]:
            mk = skeleton.mu[k]
            data_ready = max(data_ready, finish[k] + inst.data_delay[(k, u, mk, m)])
        state_ready = 0.0
        if rho is not None:
            if rho not in scheduled:
                return None
            state_ready = finish[rho] + inst.h[(rho, u, skeleton.mu[rho], m)]
        local_ready = finish[a] if a is not None else 0.0
        duration = inst.p_cold[(u, m)] if rho is None else inst.p_cold[(u, m)] - inst.b[(rho, u, m)]
        duration = max(_EPS, duration)
        cand_start = max(data_ready, state_ready, local_ready)
        cand_finish = cand_start + duration
        if s is not None and cand_finish > start[s] + _EPS:
            return None

        t_cur = max([0.0] + [finish[v] for v in scheduled])
        omega = self._unavailable_affinity(inst, g, scheduled, u, rho)
        local_bonus = inst.p_bar if (rho is not None and skeleton.mu.get(rho) == m) else 0.0
        q_reuse = 0.0
        if rho is not None:
            q_reuse = inst.b[(rho, u, m)] - inst.h[(rho, u, skeleton.mu[rho], m)]
        phi = (
            self.w_mk * max(t_cur, cand_finish)
            + self.w_dn * (cand_finish + g.tail.get(u, 0.0))
            + self.w_omega * omega
            - self.w_pr * g.a_out.get(u, 0.0)
            - self.w_lc * local_bonus
            - self.w_rs * q_reuse
        )
        # tuple ordering gives deterministic tie-breaks.
        return (phi, cand_finish, pos, m, rho, cand_start, duration)

    def _fallback_insert_score(
        self,
        inst: _Instance,
        g: _Guidance,
        skeleton: _Skeleton,
        scheduled: Set[str],
        start: Dict[str, float],
        finish: Dict[str, float],
        u: str,
    ) -> Tuple[float, float, int, int, Optional[str], float, float]:
        best = None
        for m in inst.eligible[u]:
            seq = skeleton.pi[m]
            a = seq[-1] if seq else None
            local_ready = finish[a] if a is not None else 0.0
            data_ready = 0.0
            for k in inst.preds[u]:
                data_ready = max(data_ready, finish[k] + inst.data_delay[(k, u, skeleton.mu[k], m)])
            s = max(local_ready, data_ready)
            dur = inst.p_cold[(u, m)]
            c = s + dur
            cand = (c + g.tail.get(u, 0.0), c, len(seq), m, None, s, dur)
            if best is None or cand < best:
                best = cand
        assert best is not None
        return best

    def _unavailable_affinity(
        self,
        inst: _Instance,
        g: _Guidance,
        scheduled: Set[str],
        u: str,
        rho: Optional[str],
    ) -> float:
        val = 0.0
        for r in inst.reuse_by_target.get(u, []):
            if r in scheduled or r == rho:
                continue
            max_b = max(inst.b[(r, u, m)] for m in inst.eligible[u])
            val += g.alpha.get((r, u), 0.0) * max_b
        return val

    def _refine(self, inst: _Instance, g: _Guidance, skeleton: _Skeleton, rounds: int) -> _Skeleton:
        current = skeleton.copy()
        decoded = self._decode(inst, current)
        if not decoded.feasible:
            return skeleton
        for _ in range(rounds):
            loads = {
                m: sum(decoded.duration.get(u, inst.p_cold[(u, m)]) for u in current.pi[m])
                for m in inst.executors
            }
            heavy = [m for m, _ in sorted(loads.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]
            focus: List[str] = []
            for m in heavy:
                seq = current.pi[m]
                ranked = sorted(seq, key=lambda u: (decoded.finish.get(u, 0.0) + g.tail.get(u, 0.0)), reverse=True)
                focus.extend(ranked[:6])
            focus = list(dict.fromkeys(focus))

            best_skeleton = current
            best_decoded = decoded
            improved = False
            for u in focus:
                base = current.copy()
                old_m = base.mu.get(u)
                if old_m is None:
                    continue
                if u in base.pi[old_m]:
                    base.pi[old_m].remove(u)
                base.mu.pop(u, None)
                base.rho.pop(u, None)
                reuse_modes: List[Optional[str]] = [None] + [r for r in inst.reuse_by_target.get(u, []) if r != u]
                for m in inst.eligible[u]:
                    for pos in range(len(base.pi[m]) + 1):
                        for rho in reuse_modes:
                            cand = base.copy()
                            cand.pi[m].insert(pos, u)
                            cand.mu[u] = m
                            cand.rho[u] = rho
                            cand_decoded = self._decode(inst, cand)
                            if not cand_decoded.feasible:
                                continue
                            if cand_decoded.makespan + _EPS < best_decoded.makespan:
                                best_skeleton = cand
                                best_decoded = cand_decoded
                                improved = True
            current = best_skeleton
            decoded = best_decoded
            if not improved:
                break
        return current

    # ------------------------------------------------------------------
    # Decoding / fallback / plan conversion
    # ------------------------------------------------------------------
    def _decode(self, inst: _Instance, skeleton: _Skeleton) -> _DecodeResult:
        adj: Dict[str, Set[str]] = {u: set() for u in inst.tasks}
        indeg: Dict[str, int] = {u: 0 for u in inst.tasks}

        def add_edge(a: str, b: str) -> bool:
            if a == b:
                return False
            if a not in adj or b not in adj:
                return False
            if b not in adj[a]:
                adj[a].add(b)
                indeg[b] += 1
            return True

        for u, v in inst.data_edges:
            add_edge(u, v)
        for u, rho in skeleton.rho.items():
            if rho is not None and not add_edge(rho, u):
                return _DecodeResult(False, reason=f"invalid reuse edge {rho}->{u}")
        local_pred: Dict[str, Optional[str]] = {u: None for u in inst.tasks}
        for m, seq in skeleton.pi.items():
            for i, u in enumerate(seq):
                if skeleton.mu.get(u) != m:
                    return _DecodeResult(False, reason=f"task {u} appears on mismatched executor timeline")
                if i > 0:
                    a = seq[i - 1]
                    local_pred[u] = a
                    add_edge(a, u)
        if set(skeleton.mu.keys()) != set(inst.tasks):
            return _DecodeResult(False, reason="skeleton does not assign all tasks")

        queue = [u for u in inst.topo if indeg[u] == 0]
        order: List[str] = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in sorted(adj[u], key=lambda x: inst.topo.index(x) if x in inst.topo else 10**9):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(order) != len(inst.tasks):
            return _DecodeResult(False, reason="cycle in data/reuse/local-order graph")

        start: Dict[str, float] = {}
        finish: Dict[str, float] = {}
        duration: Dict[str, float] = {}
        for u in order:
            m = skeleton.mu[u]
            if m not in inst.eligible[u]:
                return _DecodeResult(False, reason=f"task {u} assigned to ineligible executor {m}")
            rho = skeleton.rho.get(u)
            dur = inst.p_cold[(u, m)]
            state_ready = 0.0
            if rho is not None:
                if rho not in finish:
                    return _DecodeResult(False, reason=f"reuse source {rho} for {u} is not ready")
                dur -= inst.b[(rho, u, m)]
                state_ready = finish[rho] + inst.h[(rho, u, skeleton.mu[rho], m)]
            dur = max(_EPS, dur)
            data_ready = 0.0
            for k in inst.preds[u]:
                data_ready = max(data_ready, finish[k] + inst.data_delay[(k, u, skeleton.mu[k], m)])
            local_ready = 0.0
            pred = local_pred.get(u)
            if pred is not None:
                local_ready = finish[pred]
            s = max(0.0, data_ready, local_ready, state_ready)
            c = s + dur
            start[u] = s
            finish[u] = c
            duration[u] = dur
        return _DecodeResult(True, start, finish, duration, max(finish.values()) if finish else 0.0, order)

    def _cold_eft_fallback(self, inst: _Instance) -> _Skeleton:
        skeleton = _Skeleton(mu={}, rho={}, pi={m: [] for m in inst.executors})
        avail = {m: 0.0 for m in inst.executors}
        finish: Dict[str, float] = {}
        for u in inst.topo:
            best = None
            for m in inst.eligible[u]:
                data_ready = 0.0
                for k in inst.preds[u]:
                    data_ready = max(data_ready, finish[k] + inst.data_delay[(k, u, skeleton.mu[k], m)])
                start = max(avail[m], data_ready)
                end = start + inst.p_cold[(u, m)]
                cand = (end, start, m)
                if best is None or cand < best:
                    best = cand
            assert best is not None
            end, _start, m = best
            skeleton.mu[u] = m
            skeleton.rho[u] = None
            skeleton.pi[m].append(u)
            avail[m] = end
            finish[u] = end
        return skeleton

    def _to_plan(
        self,
        inst: _Instance,
        g: _Guidance,
        skeleton: _Skeleton,
        decoded: _DecodeResult,
        guidance_method: str,
        warnings: List[str],
    ) -> SchedulePlan:
        steps: Dict[str, ScheduleStep] = {}
        order_pos = {u: i for i, u in enumerate(decoded.topo or inst.topo)}
        for u in inst.tasks:
            m = skeleton.mu[u]
            rho = skeleton.rho.get(u)
            cold = inst.p_cold[(u, m)]
            dur = decoded.duration.get(u, cold)
            if rho is None:
                mode = "cold"
            else:
                src_m = skeleton.mu.get(rho)
                mode = "local" if src_m == m else "remote"
            steps[u] = ScheduleStep(
                op_id=u,
                worker_id=m,
                planned_start=decoded.start.get(u, 0.0),
                planned_end=decoded.finish.get(u, 0.0),
                estimated_duration=dur,
                cold_duration=cold,
                reuse_from=rho,
                reuse_mode=mode,
                priority=g.L.get(u, 0.0) + self.beta * g.a_out.get(u, 0.0),
                order_index=order_pos.get(u, 10**9),
            )
        return SchedulePlan(
            steps=steps,
            timelines={m: list(skeleton.pi[m]) for m in inst.executors},
            makespan=decoded.makespan,
            guidance_method=guidance_method,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _topological_order(self, tasks: Sequence[str], preds: Mapping[str, Set[str]], succs: Mapping[str, Set[str]]) -> List[str]:
        indeg = {u: len(preds[u]) for u in tasks}
        queue = sorted([u for u in tasks if indeg[u] == 0])
        out: List[str] = []
        while queue:
            u = queue.pop(0)
            out.append(u)
            for v in sorted(succs[u]):
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if len(out) != len(tasks):
            raise ValueError("Cycle detected in MFE operator DAG")
        return out

    def _descendants(self, tasks: Sequence[str], succs: Mapping[str, Set[str]]) -> Dict[str, Set[str]]:
        memo: Dict[str, Set[str]] = {}

        def dfs(u: str) -> Set[str]:
            if u in memo:
                return memo[u]
            seen: Set[str] = set()
            for v in succs[u]:
                seen.add(v)
                seen.update(dfs(v))
            memo[u] = seen
            return seen

        for u in tasks:
            dfs(u)
        return memo
