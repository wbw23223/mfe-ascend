# vLLM Ascend Docker + Volume 挂载部署手册

本文适用于：服务器已经按 vLLM Ascend quickstart 启动过 Docker，容器通过 `-v` 挂载宿主机目录，内网访问受限，不能指望容器直接访问 GitHub/Hugging Face。

目标路径约定：

```bash
export WORK=/data/mfe
export PROJECT=$WORK/mfe-ascend
export MODEL_DIR=$WORK/models/Qwen3-0.6B
export BUNDLE=$WORK/offline-bundle
```

## 1. 在可联网机器准备离线包

```bash
mkdir -p offline-bundle/{repo,models,data}
git clone https://github.com/micavro/mfe-ascend.git
tar -czf offline-bundle/repo/mfe-ascend.tar.gz mfe-ascend
```

准备一个小模型目录，推荐先用 quickstart 同级别的小模型验证链路。模型目录必须包含 `config.json`、tokenizer 文件和权重文件。

```bash
tar -czf offline-bundle/models/Qwen3-0.6B.tar.gz -C /path/to/models Qwen3-0.6B
```

准备 GSM8K 小数据。若外网机器可以跑 Python：

```bash
cd mfe-ascend
python -m pip install -e . --no-deps
python -m pip install datasets pandas pyarrow
python -m mfe.scripts.download_datasets --datasets gsm8k --limit 50
cd ..
tar -czf offline-bundle/data/mfe-data-gsm8k.tar.gz -C mfe-ascend data
```

把 `offline-bundle` 传到服务器 `/data/mfe/offline-bundle`。

## 2. 在服务器宿主机解包

```bash
mkdir -p $WORK/models
tar -xzf $BUNDLE/repo/mfe-ascend.tar.gz -C $WORK
tar -xzf $BUNDLE/data/mfe-data-gsm8k.tar.gz -C $PROJECT
tar -xzf $BUNDLE/models/Qwen3-0.6B.tar.gz -C $WORK/models

ls -lh $PROJECT
ls -lh $PROJECT/data/gsm8k/gsm8k.parquet
ls -lh $MODEL_DIR
```

## 3. 重新启动带 volume 的 vLLM Ascend 容器

沿用 quickstart 的镜像和设备挂载，只额外加 `-v /data/mfe:/data/mfe` 和 `-w /data/mfe/mfe-ascend`。

```bash
export IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1
export DEVICE=/dev/davinci0

docker run --rm -it \
  --name mfe-ascend \
  --shm-size=1g \
  --device $DEVICE \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -v /data/mfe:/data/mfe \
  -w /data/mfe/mfe-ascend \
  -p 8000:8000 \
  $IMAGE bash
```

多卡时增加 `--device /dev/davinci1` 等，并在容器内设置 `MFE_DEVICE_IDS=0,1`。

## 4. 容器内初始化

```bash
cd /data/mfe/mfe-ascend

export MFE_ACCELERATOR=ascend
export MFE_DEVICE_IDS=0
export MFE_MODEL_PATH=/data/mfe/models/Qwen3-0.6B
export MFE_DATA_DIR=/data/mfe/mfe-ascend/data
export MFE_OUTPUT_DIR=/data/mfe/mfe-ascend/data/gsm8k
export MFE_OFFLINE=1
export VLLM_TARGET_DEVICE=npu
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python -m pip install -e . --no-deps
```

不要在官方 vLLM Ascend 容器里直接 `pip install -e .`，否则可能覆盖容器内已经匹配好的 torch/torch-npu/vLLM/vLLM Ascend。

## 5. 环境体检

```bash
python -m mfe.scripts.check_ascend_env \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --device-ids "$MFE_DEVICE_IDS" \
  --offline
```

通过标准：

- `summary.ok` 为 `true`。
- `npu-smi info` 成功。
- `torch-npu` 与 `vllm-ascend` importable。
- `model_path` 和 `data/gsm8k/gsm8k.parquet` 存在。

## 6. 跑 MFE 调度链路

这一步不调用真实 vLLM/NPU，只验证代码、数据、模板和多进程调度。

```bash
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 5 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --output-dir "$MFE_OUTPUT_DIR" \
  --test-worker \
  --worker-delay 0 \
  --offline \
  -v
```

## 7. 跑 vLLM Ascend smoke test

```bash
python -m mfe.scripts.smoke_vllm \
  --model-path "$MFE_MODEL_PATH" \
  --prompt "What is 1+1? Answer briefly." \
  --max-tokens 32 \
  --device-ids "$MFE_DEVICE_IDS" \
  --offline
```

如果这一步失败，先不要跑 MFE 真实 NPU。优先检查模型路径、Ascend 环境变量、HBM 是否够、vLLM Ascend plugin 是否激活。

## 8. 跑 MFE 真实 NPU

单卡小样本：

```bash
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 2 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --output-dir "$MFE_OUTPUT_DIR" \
  --device-ids "$MFE_DEVICE_IDS" \
  --accelerator ascend \
  --offline \
  -v
```

多卡时：

```bash
export MFE_DEVICE_IDS=0,1
python -m mfe.scripts.client \
  --dataset gsm8k \
  -n 10 \
  --yaml adv_reason_3.yaml \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --device-ids "$MFE_DEVICE_IDS" \
  --accelerator ascend \
  --offline \
  -v
```

## 9. 收集结果

```bash
ls -lh $MFE_OUTPUT_DIR/*result*.json
python -m mfe.scripts.check_ascend_env \
  --model-path "$MFE_MODEL_PATH" \
  --data-dir "$MFE_DATA_DIR" \
  --device-ids "$MFE_DEVICE_IDS" \
  --offline > /data/mfe/check_ascend_env.json
```

结果 JSON 中每条样本都会包含 `run_info`，记录包版本、设备 ID、模型路径和离线模式。
