# Unified CUDA/Ascend deployment

This repository now uses one runtime control plane for CUDA GPUs and Huawei Ascend NPUs.
Use `MFE_ACCELERATOR=cuda|ascend` or the deployment profiles below.

## Profiles

```bash
# Lab NVIDIA A800 server, 10 homogeneous GPUs.
bash deploy/run_unified.sh lab-a800 \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7,8,9 \
  --expected-device-count 10 \
  --dataset gsm8k --num 20 --offline

# Company Ascend server, 8 homogeneous NPUs.
bash deploy/run_unified.sh company-ascend \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7 \
  --expected-device-count 8 \
  --dataset gsm8k --num 20 --offline
```

The script performs:

1. `python -m pip install -e . --no-deps`.
2. `python -m mfe.scripts.check_runtime_env` with the profile accelerator, optional expected card count, and homogeneous-device check.
3. `python -m mfe.scripts.client` with the same runtime settings.

It intentionally does not install or upgrade `torch`, `torch-npu`, `vllm`, or `vllm-ascend`.
Those packages must match the server image, CUDA stack, or CANN/vLLM Ascend compatibility matrix.

## Check only

Run this before real inference when validating a new machine image:

```bash
bash deploy/run_unified.sh lab-a800 \
  --mode check \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7,8,9 \
  --expected-device-count 10 \
  --offline
```

For Ascend:

```bash
bash deploy/run_unified.sh company-ascend \
  --mode check \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0,1,2,3,4,5,6,7 \
  --expected-device-count 8 \
  --offline
```

The check fails if:

- `--expected-device-count` is set and the visible card count differs from it;
- visible devices have different model names or memory sizes when those fields are inspectable;
- required runtime packages are missing;
- required data/model paths are missing.

## vLLM smoke

Before running a full MFE workflow, run one direct vLLM generation:

```bash
bash deploy/run_unified.sh company-ascend \
  --mode smoke \
  --model-path /data/mfe/models/Qwen3-0.6B \
  --device-ids 0 \
  --offline
```

For CUDA, use `lab-a800` and a CUDA-visible device ID.

## Scheduler smoke without real cards

To verify the DAG/server/scheduler path without vLLM or accelerator devices:

```bash
bash deploy/run_unified.sh custom \
  --accelerator cuda \
  --expected-device-count 0 \
  --mode test-worker \
  --dataset gsm8k --num 1 \
  --skip-install
```

This mode still reads the dataset and templates, but uses `TestWorker` instead of vLLM.
