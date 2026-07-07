# SAI-LP scheduler integration

This patch adds an admission-time SAI-LP scheduler to MFE Ascend.

## Enable it

```bash
export MFE_SCHEDULER=sailp
# Optional: ask vLLM to enable automatic prefix caching when your vLLM version supports it.
export MFE_ENABLE_PREFIX_CACHING=1
python -m mfe.scripts.client --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

The existing eager scheduler remains the default.  Use `MFE_SCHEDULER=eager` or unset the variable to return to the original behavior.

## What is implemented

The new `mfe.optimizers.sailp.SAILPScheduler` implements the SAI-LP structure:

1. It builds a WSAS instance from the YAML DAG, worker count, optional executor eligibility, estimated cold times, data-transfer delays, and optional reuse candidates.
2. It solves a reduced placement-reuse LP with SciPy when available.  If SciPy is missing or `MFE_SAILP_USE_LP=0`, it uses a HEFT-style SAI-NoLP fallback.
3. It constructs a feasible executor timeline by state-affinity insertion: for each data-ready op it jointly scores executor, insertion gap, and reuse source.
4. It runs a bounded makespan-executor refinement pass.
5. `MultiRequestOptimizer` dispatches ready ops according to the planned worker/timeline.  Set `MFE_SAILP_STRICT=1` to force only the planned worker; by default the runtime may fall back to another idle worker if the planned worker is busy.

## YAML metadata

All fields are optional.  Existing templates continue to work.

```yaml
ops:
  op1:
    model: "${MFE_MODEL_PATH}"
    input_ops: [op0]
    output_ops: [op2]

    # Optional reuse hints.  A target can choose at most one source.
    reuse_from:
      - op_id: op0
        benefit: 1.2        # estimated time saved by warm/prefix reuse
        remote_delay: 0.15  # estimated remote state-access delay

    # Optional group shorthand.  Tasks in the same group become reuse candidates
    # after causal-cycle filtering.
    reuse_group: shared_context_A

    # Optional worker compatibility.  Omit for all visible workers.
    eligible_devices: [0, 1]

    # Optional timing profile used by the planner.
    sailp:
      cold_time: 2.5
      data_delay: 0.05
      reuse_benefit: 1.0
      remote_state_delay: 0.20
      device_time:
        "0": 2.2
        "1": 2.8
```

## Useful environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MFE_SCHEDULER` | `eager` | Set to `sailp` to enable the new scheduler. |
| `MFE_ENABLE_SAILP` | `0` | Alternative boolean switch. |
| `MFE_SAILP_USE_LP` | `auto` | `auto`, `1`, or `0`; falls back if SciPy is unavailable. |
| `MFE_SAILP_STRICT` | `0` | If `1`, dispatch only to the planned worker. |
| `MFE_SAILP_REFINEMENT_ROUNDS` | `2` | Bounded bottleneck-refinement rounds. |
| `MFE_SAILP_DEFAULT_OP_TIME` | `1.0` | Base cold-time estimate when YAML lacks timing. |
| `MFE_SAILP_TOKEN_TIME` | `0.001` | Added per `max_tokens` to estimate cold time. |
| `MFE_SAILP_REUSE_BENEFIT_RATIO` | `0.35` | Default warm-reuse benefit as a fraction of cold time. |
| `MFE_SAILP_CROSS_DATA_DELAY` | `0.05` | Default cross-worker data delay. |
| `MFE_SAILP_REMOTE_STATE_DELAY` | `0.10` | Default remote state-access delay. |
| `MFE_SAILP_DEVICE_SPEEDS` | all `1.0` | Worker speed hints, e.g. `0:1.0,1:0.8`. |

## Runtime status output

`status(uid)` now includes:

```json
{
  "scheduler": "sailp",
  "schedule_plan": {
    "makespan": 12.3,
    "guidance_method": "lp",
    "timelines": {"0": ["op0", "op1"], "1": ["op2"]},
    "steps": {
      "op1": {
        "worker_id": 0,
        "reuse_from": "op0",
        "reuse_mode": "local",
        "planned_start": 3.0,
        "planned_end": 4.5
      }
    }
  }
}
```

## Limitations

The planner is admission-time and estimates processing/reuse costs from YAML or environment defaults.  It does not directly move KV-cache between workers.  Real cache benefits require the backend to support prefix/KV reuse, such as vLLM automatic prefix caching on the same worker.  Remote reuse is modeled for placement decisions; implementing explicit remote KV transfer would require deeper vLLM/cache-manager APIs.
