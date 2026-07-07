# openEuler/Ascend 迁移说明

本文档记录 `mfe-ascend` 在华为昇腾服务器上的首轮落地步骤。当前还不知道目标机器的卡数、HBM、CANN 版本和驱动版本，所以代码侧只做运行时探测，不写死设备数量。

## 目标环境

- OS: openEuler Linux
- Accelerator: Huawei Ascend NPU
- Runtime: CANN + torch-npu + vllm-ascend
- Python: 3.10 或 3.11
- 默认安装不覆盖 `torch`、`torch-npu`、`vllm`、`vllm-ascend`；如需旧版组合，参考 `constraints/ascend-legacy-vllm09.txt`。

如果服务器预装的是 CANN 8.5/9.0 或更新的 vLLM Ascend 镜像，应优先复用镜像内已经验证过的整套 `vllm`、`vllm-ascend`、`torch`、`torch-npu`，不要只改其中一个包。

## 上机后先检查

```bash
npu-smi info
python -m mfe.scripts.check_ascend_env
```

重点确认：

- `/etc/os-release` 是 openEuler。
- `npu-smi info` 能看到卡。
- `torch_npu` 可 import。
- `torch.npu.device_count()` 大于 0。
- `vllm` 与 `vllm_ascend` 版本处在同一兼容矩阵。

## 安装

推荐先使用官方 vLLM Ascend openEuler 容器或服务器预装环境。裸机安装时，需要先由管理员安装 Ascend driver/firmware、CANN 和相关系统库。

```bash
cd /path/to/mfe-ascend
python -m pip install -U pip
python -m pip install -e . --no-deps
```

如果服务器已使用官方 vLLM Ascend 容器，容器内可能已装好 `torch`、`torch-npu`、`vllm`、`vllm-ascend`。这种情况下可以先检查版本，再决定是否使用 `--no-deps`：

```bash
python -m pip install -e . --no-deps
python -m mfe.scripts.check_ascend_env
```

## 运行

先设置可见 NPU。设备 ID 以 `npu-smi info` 为准。

```bash
source .env.ascend.example
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPU_VISIBLE_DEVICES=0,1
```

模板里的 `model` 需要改成服务器上真实可访问的模型路径。例如：

```yaml
model: "${MFE_MODEL_PATH}"
```

下载或准备数据后运行：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 20 --yaml adv_reason_3.yaml --send-interval 0.0 -v
```

无真实 NPU 时可先跑调度链路：

```bash
python -m mfe.scripts.client --dataset gsm8k -n 5 --yaml adv_reason_3.yaml --test-worker --worker-delay 0.2 -v
```

## 代码迁移点

- `mfe.util` 负责 `ascend/cuda` 后端选择、设备枚举和 worker 设备绑定。
- `mfe.optimizers.multi_request.MultiRequestOptimizer` 不再直接调用 `torch.cuda.device_count()`。
- `mfe.workers.worker_v.vLLMWorker` 在导入 vLLM 前设置 `ASCEND_RT_VISIBLE_DEVICES`、`VLLM_TARGET_DEVICE=npu` 和 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。
- 项目已整理为正常包结构：源码位于 `mfe/`，命令使用 `python -m mfe.scripts...`。

## 未知项

等看到机器后需要补齐：

- Ascend 型号、卡数、单卡 HBM。
- CANN、driver、firmware 版本。
- 是否使用官方容器，容器镜像 tag。
- 目标模型路径、模型架构、上下文长度、量化方式。
- 单 worker 单卡是否足够，是否需要 vLLM tensor parallel/pipeline parallel。
