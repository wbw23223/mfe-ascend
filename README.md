# MFE Ascend + SAI-LP

MFE Ascend 是基于 `micavro/mfe` 的昇腾迁移版本。它保留原 MFE 的 YAML DAG 工作流、多请求调度、submit/status 查询和 benchmark 输出，并将默认推理后端迁移到 openEuler + Huawei Ascend NPU + vLLM Ascend。

本 fork 在原有 eager ready-task 调度器之外，新增了 **SAI-LP（LP-guided State-Affinity Insertion Scheduling）状态亲和调度器**。SAI-LP 面向多 Agent / 多步骤 LLM 工作流，在 admission-time 根据 DAG、executor 兼容性、估计执行时间、数据传输代价和可选 KV-cache / prefix state 复用关系生成调度计划。

> 默认行为仍保持原仓库的 eager 调度；只有设置 `MFE_SCHEDULER=sailp` 或 `MFE_ENABLE_SAILP=1` 后才启用 SAI-LP。

## 核心特性

- **Ascend 优先的多设备执行**：默认面向 Ascend NPU，可通过 `MFE_ACCELERATOR=ascend|cuda|auto` 切换。
- **YAML DAG 工作流**：每个 workflow 由 `templates/*.yaml` 描述，支持多阶段推理、并行分支、聚合、投票等结构。
- **多请求调度**：server 接收 submit/status 请求，后台 scheduler 将 ready op 分派给 worker。
- **vLLM / vLLM Ascend Worker**：真实 worker 使用 vLLM 执行模型推理，测试 worker 可在无 NPU 环境下验证调度流程。
- **SAI-LP 状态亲和调度**：新增 admission-time planner，联合考虑任务关键路径、executor timeline、reuse source、local/remote state access 代价。
- **可选 prefix caching**：设置 `MFE_ENABLE_PREFIX_CACHING=1` 后，worker 会尝试启用 vLLM automatic prefix caching；实际支持情况取决于当前 vLLM/vLLM Ascend 版本。

## SAI-LP 是什么

SAI-LP 来自论文 **SAIL: State-Affinity Insertion with LP Guidance for Multi-Agent LLM Workflow Scheduling**。论文将多 Agent LLM workflow 建模为状态感知异构 DAG 调度问题：每个任务不仅要决定在哪个 executor 上运行，还要决定是否复用上游任务产生的 KV-cache / prefix state，以及本地复用还是远程访问。

SAI-LP 的主要流程：

1. **LP guidance**：先求一个简化的 placement-reuse LP，得到任务 criticality 和 reuse affinity。
2. **State-affinity insertion**：对每个 data-ready task，联合枚举 executor、timeline insertion gap 和 reuse mode。
3. **Makespan-executor refinement**：针对当前瓶颈 executor 做有限局部重插入，尝试降低 makespan。

在本仓库中，SAI-LP 位于 `mfe/optimizers/sailp.py`。它输出一个 `schedule_plan`，再由 `MultiRequestOptimizer` 按计划向 worker 派发 op。

## 目录结构

```text
mfe-ascend/
├── mfe/
│   ├── components/              # Operator、Query、ExecuteInfo、Benchmark 等核心对象
│   ├── optimizers/
│   │   ├── multi_request.py      # 多请求调度入口；支持 eager / sailp
│   │   └── sailp.py              # SAI-LP 状态亲和调度器
│   ├── serve/                    # submit/status server
│   ├── workers/                  # vLLM Worker / TestWorker
│   ├── scripts/                  # client、benchmark、环境检查、数据下载脚本
│   ├── parser.py                 # YAML DAG 与 SAI-LP 元数据解析
│   ├── util.py                   # Ascend/CUDA 设备发现和环境绑定
│   └── config.py
├── templates/                    # YAML 工作流模板
│   └── sailp_example.yaml        # 带 SAI-LP metadata 的示例 workflow
├── docs/
│   ├── sailp.md                  # SAI-LP 使用说明
│   ├── openeuler-ascend.md       # openEuler/Ascend 上机说明
│   └── ...
├── data/                         # 默认数据目录
├── constraints/                  # 依赖版本约束记录
├── .env.ascend.example           # Ascend 环境变量模板
├── README_SAILP.md               # SAI-LP patch 简要说明
├── README.md
└── pyproject.toml
```

## 环境要求

目标 Ascend 环境应先具备：

- openEuler Linux 或兼容 Linux 环境
- Ascend driver / firmware
- CANN
- Python 3.10 或 3.11
- torch-npu
- vLLM 与 vLLM Ascend

上机后建议先检查：

```bash
npu-smi info
python -m mfe.scripts.check_ascend_env
```

如果你使用官方 vLLM Ascend 容器，并通过 `-v` 挂载宿主机目录，优先参考：

```text
docs/vllm-ascend-docker-volume.md
```

其他部署文档：

```text
docs/ubuntu22-ascend-deploy.md
docs/openeuler-ascend.md
```

## 安装

```bash
cd /path/to/mfe-ascend
python -m pip install -U pip
python -m pip install -e . --no-deps
```

如果容器中已经预装 torch、torch-npu、vLLm、vllm-ascend，不建议让 pip 重新解析和覆盖这些依赖。

## 数据目录

默认数据根目录是项目根目录下的 `data/`：

```text
mfe-ascend/data/
```

推荐目录结构：

```text
mfe-ascend/data/gsm8k/gsm8k.parquet
mfe-ascend/data/drop/drop.parquet
mfe-ascend/data/hotpotqa/hotpotqa.parquet
mfe-ascend/data/math/math.parquet
```

目录名和 parquet 文件名建议使用小写。命令行中的 `--dataset GSM8k` 会被归一化为 `gsm8k`，但实际文件路径仍应是：

```text
data/gsm8k/gsm8k.parquet
```

下载 GSM8K 小样本：

```bash
cd /path/to/mfe-ascend
python -m mfe.scripts.download_datasets --datasets gsm8k --limit 50 --data-dir data
```

如果数据已由他人提供，直接复制到对应目录：

```bash
mkdir -p data/gsm8k
cp /path/to/gsm8k.parquet data/gsm8k/gsm8k.parquet
```

运行时可显式指定数据目录：

```bash
export MFE_DATA_DIR=$PWD/data
python -m mfe.scripts.client \
  --dataset gsm8k \
  --data-dir "$MFE_DATA_DIR" \
  -n 5 \
  --test-worker \
  --worker-delay 0
```

## 基础运行

先按机器实际卡号设置环境变量：

```bash
source .env.ascend.example
export MFE_DEVICE_IDS=0,1
export MFE_MODEL_PATH=/data/mfe/models/Qwen3-0.6B
export MFE_DATA_DIR=$PWD/data
export MFE_OUTPUT_DIR=$PWD/data/gsm8k
```

确认模板里的模型路径使用环境变量：

```yaml
model: "${MFE_MODEL_PATH}"
```

运行真实 vLLM / Ascend worker：

```bash
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 20 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --offline \
  -v
```

无 NPU 或只想验证调度流程时，使用 TestWorker：

```bash
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 5 \
  --yaml adv_reason_3.yaml \
  --test-worker \
  --worker-delay 0.2 \
  -v
```

## 启用 SAI-LP 调度器

默认 scheduler 是 eager。启用 SAI-LP：

```bash
export MFE_SCHEDULER=sailp
```

或者：

```bash
export MFE_ENABLE_SAILP=1
```

如果当前 vLLM 版本支持 automatic prefix caching，可同时打开：

```bash
export MFE_ENABLE_PREFIX_CACHING=1
```

使用测试 worker 验证：

```bash
export MFE_SCHEDULER=sailp
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 5 \
  --yaml sailp_example.yaml \
  --test-worker \
  --worker-delay 0.2 \
  -v
```

使用真实 Ascend worker：

```bash
export MFE_SCHEDULER=sailp
export MFE_ENABLE_PREFIX_CACHING=1
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 20 \
  --yaml sailp_example.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --offline \
  -v
```

如果需要严格按照 SAI-LP 规划的 worker 派发，而不是在计划 worker 繁忙时回退到其他空闲 worker：

```bash
export MFE_SAILP_STRICT=1
```

## SAI-LP YAML 元数据

原有 YAML 模板不需要修改也可以运行 SAI-LP。没有显式 metadata 时，planner 会从环境变量和 op 参数估计 cold time、data delay、reuse benefit 等。

为了让 SAI-LP 更准确地做状态亲和调度，可以在 `templates/*.yaml` 中添加可选字段：

```yaml
ops:
  op1:
    model: "${MFE_MODEL_PATH}"
    prompt: "..."
    max_tokens: 512
    input_ops: [op0]
    output_ops: [op2]

    # 可选：显式状态复用候选。target op 至多选择一个 reuse source。
    reuse_from:
      - op_id: op0
        benefit: 1.2        # warm/prefix reuse 预计节省的时间
        remote_delay: 0.15  # 跨 worker 访问 state 的预计延迟

    # 可选：同组任务会自动生成 reuse candidates，并过滤因果环。
    reuse_group: shared_context_A

    # 可选：该 op 可运行的 worker / device。省略则表示所有可见 worker。
    eligible_devices: [0, 1]

    # 可选：planner 使用的 timing profile。
    sailp:
      cold_time: 2.5
      data_delay: 0.05
      reuse_benefit: 1.0
      remote_state_delay: 0.20
      device_time:
        "0": 2.2
        "1": 2.8
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `reuse_from` | 显式复用候选，表示当前 op 可以尝试复用某个上游 op 的 state。 |
| `reuse_group` | 同组 op 自动生成候选复用边，适合共享长上下文 / 共享 prefix 的 workflow。 |
| `eligible_devices` | 限制 op 可运行的 worker/device。 |
| `sailp.cold_time` | 当前 op 的 cold execution time 估计。 |
| `sailp.data_delay` | 跨 worker 数据依赖传输延迟估计。 |
| `sailp.reuse_benefit` | 默认复用收益估计。 |
| `sailp.remote_state_delay` | 跨 worker 状态访问延迟估计。 |
| `sailp.device_time` | 为不同 worker 指定不同执行时间，用于表达 executor 异构性。 |

## SAI-LP 环境变量

| 变量 | 默认值 | 作用 |
| --- | --- | --- |
| `MFE_SCHEDULER` | `eager` | 设置为 `sailp` 启用 SAI-LP。 |
| `MFE_ENABLE_SAILP` | `0` | 备用布尔开关。 |
| `MFE_SAILP_USE_LP` | `auto` | `auto`、`1` 或 `0`；SciPy 不可用时自动降级。 |
| `MFE_SAILP_STRICT` | `0` | 为 `1` 时只派发到计划 worker。 |
| `MFE_SAILP_REFINEMENT_ROUNDS` | `2` | bottleneck refinement 轮数。 |
| `MFE_SAILP_DEFAULT_OP_TIME` | `1.0` | YAML 缺少 timing 时的默认 cold time。 |
| `MFE_SAILP_TOKEN_TIME` | `0.001` | 根据 `max_tokens` 追加的 per-token time 估计。 |
| `MFE_SAILP_REUSE_BENEFIT_RATIO` | `0.35` | 默认复用收益占 cold time 的比例。 |
| `MFE_SAILP_CROSS_DATA_DELAY` | `0.05` | 默认跨 worker 数据传输延迟。 |
| `MFE_SAILP_REMOTE_STATE_DELAY` | `0.10` | 默认远程 state access 延迟。 |
| `MFE_SAILP_DEVICE_SPEEDS` | 全部 `1.0` | worker 速度提示，例如 `0:1.0,1:0.8`。 |
| `MFE_ENABLE_PREFIX_CACHING` | 未设置 | 请求 worker 启用 vLLM automatic prefix caching。 |

可选 scoring 权重：

```bash
export MFE_SAILP_W_MK=1.0
export MFE_SAILP_W_DN=0.3
export MFE_SAILP_W_OMEGA=1.5
export MFE_SAILP_W_PR=0.3
export MFE_SAILP_W_LC=0.15
export MFE_SAILP_W_RS=0.3
export MFE_SAILP_BETA=0.20
export MFE_SAILP_TAU=0.25
```

## status 输出

启用 SAI-LP 后，`status(uid)` 会包含调度计划信息：

```json
{
  "scheduler": "sailp",
  "schedule_plan": {
    "makespan": 12.3,
    "guidance_method": "lp",
    "timelines": {
      "0": ["op0", "op1"],
      "1": ["op2"]
    },
    "steps": {
      "op1": {
        "op_id": "op1",
        "worker_id": 0,
        "planned_start": 3.0,
        "planned_end": 4.5,
        "estimated_duration": 1.5,
        "cold_duration": 2.5,
        "reuse_from": "op0",
        "reuse_mode": "local",
        "priority": 4.2,
        "order_index": 1
      }
    },
    "warnings": []
  }
}
```

`guidance_method` 可能是：

- `lp`：成功使用 SciPy 求 LP guidance。
- `heuristic`：未使用 LP，采用 HEFT-style / SAI-NoLP fallback。
- `lp+fallback` 或 `heuristic+fallback`：planner 结果不可用时回退到 cold EFT fallback。

## 常用调试命令

语法检查：

```bash
python -m py_compile \
  mfe/optimizers/sailp.py \
  mfe/optimizers/multi_request.py \
  mfe/parser.py \
  mfe/components/operator.py \
  mfe/components/model_config.py \
  mfe/components/query.py \
  mfe/workers/worker_v.py
```

导入检查：

```bash
python - <<'PY'
from mfe.optimizers.sailp import SAILPScheduler
print("SAILPScheduler import OK")
PY
```

查看当前 Git 改动：

```bash
git status
git diff --stat
```

提交修改：

```bash
git add README.md docs/sailp.md README_SAILP.md templates/sailp_example.yaml mfe/
git commit -m "Add SAI-LP state-affinity scheduler"
git push -u origin feature/sailp-scheduler
```

## 设计边界与限制

- SAI-LP 是 **admission-time planner**，不是 token-level 或 iteration-level scheduler。
- 目前实现重点是调度层面的 state-affinity placement：决定 op 放在哪个 worker、timeline 顺序如何、是否应当靠近某个 reuse source。
- 当前补丁不会直接跨 worker 迁移真实 KV-cache。远程 state access 目前是调度代价模型；若要真正移动 KV-cache，需要深入接入 vLLM/cache manager API。
- 本地 prefix/KV 复用收益依赖后端是否支持 automatic prefix caching，以及 prompt/template 是否真的存在可复用前缀。
- YAML 中的 timing/reuse 参数是估计值。生产环境中建议用 benchmark 日志持续校准 `cold_time`、`reuse_benefit`、`remote_state_delay` 和 `device_time`。

## 注意事项

- 不要随意单独升级 `vllm`、`torch` 或 `torch-npu`。Ascend 环境通常要求 CANN、torch、torch-npu、vLLM、vLLM Ascend 成套匹配。
- 卡数和显存不需要写死在代码里；框架会从 `MFE_DEVICE_IDS`、`ASCEND_RT_VISIBLE_DEVICES`、`NPU_VISIBLE_DEVICES` 或 torch 设备接口获取。
- 如果 Llama-3.1 等长上下文模型报 KV cache 不足，可限制上下文长度：

```bash
export MFE_MAX_MODEL_LEN=4096
```

也可以写入 YAML：

```yaml
max_model_len: 4096
gpu_memory_utilization: 0.95
```

## License

请沿用上游仓库 license。若向上游提交 PR，请在 PR 描述中说明本 fork 新增了 SAI-LP admission-time scheduler，以及该实现目前不包含跨 worker 真实 KV-cache 迁移。
