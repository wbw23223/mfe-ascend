# Ubuntu 22.04 + Ascend 部署与试运行手册

本文面向只有命令行、公司内网只能访问有限网站的服务器环境。目标是把 `mfe-ascend` 部署到 Ubuntu 22.04 昇腾服务器上，并按从低风险到真实推理的顺序跑通。

推荐顺序：

1. 采集机器信息。
2. 确认 Ascend 驱动、固件、CANN 是否已由管理员装好。
3. 优先使用 vLLM Ascend/CANN 容器；如果不能用容器，再走裸机 venv。
4. 先跑 `check_ascend_env`。
5. 再跑 MFE `--test-worker` 调度链路。
6. 最后跑 vLLM 单模型和 MFE 真实 NPU 流程。

## 0. 约定路径

以下命令假设：

```bash
export WORK=/data/mfe
export PROJECT=$WORK/mfe-ascend
export BUNDLE=$WORK/offline-bundle
export MODEL_DIR=/data/mfe/models/Qwen3-0.6B
```

没有 `/data` 权限时改成自己的目录，例如 `$HOME/mfe`。

## 1. 先采集服务器信息

登录服务器后先执行：

```bash
mkdir -p $WORK/logs

date | tee $WORK/logs/00-date.txt
uname -a | tee $WORK/logs/01-uname.txt
cat /etc/os-release | tee $WORK/logs/02-os-release.txt
lscpu | tee $WORK/logs/03-lscpu.txt
free -h | tee $WORK/logs/04-mem.txt
df -h | tee $WORK/logs/05-disk.txt

which npu-smi || true
npu-smi info | tee $WORK/logs/06-npu-smi.txt

which docker || true
docker --version || true
```

确认点：

- `/etc/os-release` 是 Ubuntu 22.04。
- `npu-smi info` 能看到 Ascend NPU。
- 磁盘至少预留模型大小 + 镜像/环境大小，建议先留 200GB 以上。
- 如果 `npu-smi` 不存在或报错，先找管理员处理驱动/固件/CANN，MFE 暂时不用继续装。

## 2. 部署策略选择

优先级：

1. **容器优先**：更容易复现，适合内网机器。外网机器拉好镜像后 `docker save`，传到内网 `docker load`。
2. **裸机 venv**：适合服务器已经装好 CANN，并且公司内网有可用 pip 源或 wheel 包。

vLLM Ascend 官方安装文档要求 Linux、Python、Ascend NPU，并强调 CANN、torch、torch-npu、vLLM、vLLM Ascend 要按兼容矩阵成套匹配；官方也提供 pip 与 Docker 两种安装方式。不要只升级其中一个 Python 包。

## 3. 准备离线材料包

在能访问外网的机器上准备：

```bash
mkdir -p offline-bundle/{repo,wheels,models,docker,logs}
```

### 3.1 代码包

如果服务器能访问 GitHub：

```bash
git clone https://github.com/micavro/mfe-ascend.git
```

如果服务器不能访问 GitHub，在外网机器上：

```bash
git clone https://github.com/micavro/mfe-ascend.git
tar -czf offline-bundle/repo/mfe-ascend.tar.gz mfe-ascend
```

传到服务器后：

```bash
mkdir -p $WORK
tar -xzf $BUNDLE/repo/mfe-ascend.tar.gz -C $WORK
```

### 3.2 模型包

内网受限时，不要让服务器运行时再连 Hugging Face。提前在外网机器下载模型，然后打包。

推荐保存完整 Hugging Face 目录，包含：

```text
config.json
tokenizer.json / tokenizer.model
tokenizer_config.json
generation_config.json
*.safetensors
```

打包：

```bash
tar -czf offline-bundle/models/Qwen3-0.6B.tar.gz -C /path/to/models Qwen3-0.6B
```

服务器解包：

```bash
mkdir -p /data/mfe/models
tar -xzf $BUNDLE/models/Qwen3-0.6B.tar.gz -C /data/mfe/models
```

### 3.3 数据包

如果服务器不能访问 Hugging Face datasets，在外网机器准备 `data/gsm8k/gsm8k.parquet` 等数据：

```bash
cd mfe-ascend
python -m mfe.scripts.download_datasets --datasets gsm8k --limit 100
tar -czf ../offline-bundle/repo/mfe-data-gsm8k.tar.gz data
```

服务器解包：

```bash
tar -xzf $BUNDLE/repo/mfe-data-gsm8k.tar.gz -C $PROJECT
```

### 3.4 Python wheel 包

如果服务器能访问公司 PyPI 镜像，可以跳过本节。否则在外网 Linux 机器上准备 wheelhouse。

注意：wheel 必须匹配服务器架构。先在服务器执行 `uname -m`。常见是 `x86_64` 或 `aarch64`。不同架构不要混用。

建议先按项目当前固定版本下载：

```bash
mkdir -p offline-bundle/wheels
python3.10 -m pip download -d offline-bundle/wheels \
  -r <(python3.10 - <<'PY'
import tomllib
deps = tomllib.load(open("mfe-ascend/pyproject.toml", "rb"))["project"]["dependencies"]
print("\n".join(deps))
PY
)
```

如果公司允许访问内部 PyPI，推荐把这些 wheel 上传到内网制品库；服务器只从内网源安装。

### 3.5 Docker 镜像包

如果允许 Docker，优先准备容器镜像。可以选择：

- vLLM Ascend 预构建镜像。
- CANN Ubuntu 22.04 镜像 + 在容器内安装 MFE。
- 公司内部已经验证过的 Ascend PyTorch/vLLM 镜像。

外网机器：

```bash
docker pull <IMAGE>
docker save <IMAGE> -o offline-bundle/docker/vllm-ascend.tar
```

服务器：

```bash
docker load -i $BUNDLE/docker/vllm-ascend.tar
docker images | grep -E 'vllm|ascend|cann'
```

## 4. 服务器目录布置

```bash
mkdir -p $WORK
cd $WORK

# 如果 GitHub 可访问
git clone https://github.com/micavro/mfe-ascend.git

# 如果使用离线包
tar -xzf $BUNDLE/repo/mfe-ascend.tar.gz -C $WORK

cd $PROJECT
```

## 5. 方案 A：容器部署

容器方式仍要求宿主机已经安装 Ascend 驱动，并能正常运行 `npu-smi info`。

### 5.1 找到 NPU 设备文件

```bash
ls -l /dev/davinci* /dev/devmm_svm /dev/hisi_hdc 2>/dev/null || true
ls -l /usr/local/bin/npu-smi /usr/local/dcmi /usr/local/Ascend/driver 2>/dev/null || true
```

如果要暴露 0、1 两张卡，通常需要挂载：

```text
/dev/davinci0
/dev/davinci1
/dev/davinci_manager
/dev/devmm_svm
/dev/hisi_hdc
```

### 5.2 启动容器

把 `<IMAGE>` 换成实际镜像名：

```bash
export IMAGE=<IMAGE>

docker run --rm -it \
  --name mfe-ascend \
  --network host \
  --ipc host \
  --device /dev/davinci0 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /data:/data \
  -w $PROJECT \
  $IMAGE bash
```

多卡时继续添加 `--device /dev/davinci1` 等。不要一开始暴露所有卡，先单卡跑通。

### 5.3 容器内初始化

```bash
cd $PROJECT

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi

export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS=0
export ASCEND_RT_VISIBLE_DEVICES=0
export NPU_VISIBLE_DEVICES=0
export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 5.4 安装 MFE

如果镜像内已经装好 torch/torch-npu/vllm/vllm-ascend：

```bash
python -m pip install -e . --no-deps
```

如果镜像内没装，并且容器能访问内网 pip 源：

```bash
python -m pip install -U pip
python -m pip install -e . --no-deps
```

如果完全离线：

```bash
python -m pip install --no-index --find-links $BUNDLE/wheels -e .
```

## 6. 方案 B：裸机 venv 部署

只有在宿主机 CANN 已正确安装时使用。

### 6.1 系统依赖

需要管理员安装基础包。Ubuntu 22.04 示例：

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10 python3.10-venv python3.10-dev \
  gcc g++ cmake make git curl wget jq libnuma-dev
```

如果内网不能访问 apt 源，需要公司提供 Ubuntu 22.04 内网 apt 镜像或 deb 包。

### 6.2 CANN 环境变量

```bash
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
  echo "CANN set_env.sh not found"
fi

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
  source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

建议把上面两段追加到 `~/.bashrc`，但第一次先手动 source，便于排查。

### 6.3 创建 venv

```bash
cd $PROJECT
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 6.4 安装依赖

内网 pip 源可用：

```bash
python -m pip install -e . --no-deps
```

完全离线：

```bash
python -m pip install --no-index --find-links $BUNDLE/wheels -e .
```

如果 `torch`、`torch-npu`、`vllm`、`vllm-ascend` 是管理员预装的系统环境，使用：

```bash
python -m pip install -e . --no-deps
```

然后逐个检查：

```bash
python - <<'PY'
import torch
import torch_npu
import vllm
import vllm_ascend
print("torch", torch.__version__)
print("npu count", torch.npu.device_count())
print("vllm", vllm.__version__)
print("vllm_ascend imported")
PY
```

## 7. 环境体检

```bash
cd $PROJECT
python -m mfe.scripts.check_ascend_env | tee $WORK/logs/10-check-ascend-env.json
```

必须看到：

- `npu-smi info` 成功。
- `torch-npu` installed/importable。
- `vllm-ascend` installed/importable。
- `torch_npu.device_count` 大于 0。

如果 `vllm_ascend` import 失败，先不要跑 MFE，优先处理版本匹配。

## 8. 先跑 MFE 调度链路

这一步不需要真实 NPU，用来验证项目路径、数据、模板、server/optimizer/client 逻辑。

准备数据：

```bash
ls -lh $PROJECT/data/gsm8k/gsm8k.parquet
```

如果没有数据，先从离线包解压，或在可联网环境执行：

```bash
python -m mfe.scripts.download_datasets --datasets gsm8k --limit 20
```

跑调度链路：

```bash
cd $PROJECT
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 5 \
  --yaml adv_reason_3.yaml \
  --test-worker \
  --worker-delay 0.2 \
  -v | tee $WORK/logs/20-mfe-test-worker.log
```

成功标志：

- 看到 `[REQ]`、`[SERVER]`、`[OPT]`、`[Worker]` 日志。
- 结果写入 `data/gsm8k/gsm8k_adv_reason_3_result_5.json`。

## 9. 修改模板模型路径

运行时通过 `MFE_MODEL_PATH` 或 `--model-path` 覆盖模板中的模型路径：

```bash
export MFE_MODEL_PATH=$MODEL_DIR
grep -R "model:" -n templates
```

例如：

```bash
python -m mfe.scripts.check_ascend_env --model-path "$MFE_MODEL_PATH" --data-dir "$PROJECT/data"
```

确认：

```bash
grep -n "model:" templates/adv_reason_3.yaml
ls -lh $MODEL_DIR
```

## 10. 先跑 vLLM Ascend 最小脚本

在跑 MFE 之前，先确认 vLLM Ascend 能单独推理：

```bash
cd $PROJECT
export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS=0
export MFE_MODEL_PATH=$MODEL_DIR

python -m mfe.scripts.smoke_vllm \
  --model-path "$MFE_MODEL_PATH" \
  --prompt "What is 1+1? Answer briefly." \
  --max-tokens 32 \
  --device-ids "$MFE_DEVICE_IDS" \
  --offline | tee $WORK/logs/30-vllm-smoke.log
```

如果这里失败，不要继续跑 MFE。先根据报错处理：

- 找不到模型文件：检查 `$MODEL_DIR`。
- CANN/torch-npu 动态库错误：检查 `set_env.sh`、`LD_LIBRARY_PATH`。
- OOM/HBM 不足：换小模型，或降低 `max_model_len`。
- vLLM/vllm-ascend 版本冲突：回到依赖矩阵重新安装。

## 11. 跑 MFE 真实 NPU

先单卡、小样本：

```bash
cd $PROJECT
export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS=0
export ASCEND_RT_VISIBLE_DEVICES=0
export NPU_VISIBLE_DEVICES=0
export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 2 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --device-ids "$MFE_DEVICE_IDS" \
  --send-interval 0.0 \
  -v | tee $WORK/logs/40-mfe-real-npu-1card.log
```

成功后再扩大到多卡：

```bash
export MFE_DEVICE_IDS=0,1
export ASCEND_RT_VISIBLE_DEVICES=0,1
export NPU_VISIBLE_DEVICES=0,1

python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 10 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --device-ids "$MFE_DEVICE_IDS" \
  --send-interval 0.0 \
  -v | tee $WORK/logs/41-mfe-real-npu-2card.log
```

当前 MFE 的调度策略是“一个 worker 绑定一张可见设备”。多卡时会启动多个 worker。先小批量观察 HBM，再加大 `-n`。

## 12. 常用排障命令

### 12.1 看 NPU 状态

```bash
npu-smi info
watch -n 1 npu-smi info
```

### 12.2 看 Python 包版本

```bash
python - <<'PY'
import importlib.metadata as m
for p in ["torch", "torch-npu", "vllm", "vllm-ascend", "transformers"]:
    try:
        print(p, m.version(p))
    except Exception as e:
        print(p, "ERR", e)
PY
```

### 12.3 看 Ascend 环境变量

```bash
env | grep -E 'ASCEND|NPU|VLLM|LD_LIBRARY_PATH|PYTHONPATH' | sort
```

### 12.4 完全离线模式

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 12.5 清理残留进程

```bash
ps -ef | grep -E 'mfe|vllm|python' | grep -v grep
```

确认无误后再 kill 对应 PID。

## 13. 验收清单

记录并归档这些文件：

```text
$WORK/logs/02-os-release.txt
$WORK/logs/06-npu-smi.txt
$WORK/logs/10-check-ascend-env.json
$WORK/logs/20-mfe-test-worker.log
$WORK/logs/30-vllm-smoke.log
$WORK/logs/40-mfe-real-npu-1card.log
$PROJECT/data/gsm8k/*result*.json
```

最低验收标准：

- `npu-smi info` 正常。
- `python -m mfe.scripts.check_ascend_env` 正常。
- `--test-worker` 跑通。
- vLLM Ascend smoke test 跑通。
- MFE 真实 NPU `-n 2` 跑通并生成 JSON 结果。

## 14. 内网受限时的推荐交付物

建议做一个完整离线交付目录：

```text
offline-bundle/
├── docker/
│   └── vllm-ascend.tar
├── models/
│   └── Qwen3-0.6B.tar.gz
├── repo/
│   ├── mfe-ascend.tar.gz
│   └── mfe-data-gsm8k.tar.gz
├── wheels/
│   ├── *.whl
│   └── ...
└── MANIFEST.txt
```

`MANIFEST.txt` 至少写清：

- 服务器架构：`x86_64` 或 `aarch64`。
- CANN 版本。
- torch/torch-npu/vllm/vllm-ascend 版本。
- 镜像名和 digest。
- 模型来源、模型版本、文件 hash。

## 15. 版本策略

当前仓库的 `pyproject.toml` 默认不安装或覆盖 `torch`、`torch-npu`、`vllm`、`vllm-ascend`。旧版 vLLM 0.9 组合记录在 `constraints/ascend-legacy-vllm09.txt`。

如果服务器或公司镜像已经是更新 CANN，例如 CANN 9.0，那么建议改用镜像内已验证的一整套版本，并用：

```bash
python -m pip install -e . --no-deps
```

避免 `pip install -e .` 把镜像内正确版本覆盖掉。

## 参考

- vLLM Ascend 文档说明它是 vLLM 在 Ascend NPU 上的推荐硬件插件。
- vLLM Ascend 安装文档列出 Linux、Python、Ascend NPU、CANN、torch-npu、torch 等要求，并提供 pip 和 Docker 两种安装方式。
- vLLM Ascend quickstart 使用容器方式演示单卡离线推理。
- Ascend 官方 CANN container image 仓库包含 Ubuntu 22.04 + CANN 9.0.0 + Python 3.11 的 Dockerfile 示例。
