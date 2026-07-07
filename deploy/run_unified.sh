#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash deploy/run_unified.sh <profile> [options]

Profiles:
  lab-a800           CUDA / NVIDIA A800
  company-ascend     Huawei Ascend
  custom             Use --accelerator and optional --expected-device-count

Options:
  --accelerator cuda|ascend
  --expected-device-count N
  --device-ids 0,1,2
  --model-path PATH
  --data-dir PATH
  --output-dir PATH
  --dataset NAME              default: gsm8k
  --yaml FILE                 default: adv_reason_3.yaml
  --num N                     default: 20
  --templates-dir PATH        default: templates
  --max-model-len N
  --gpu-memory-utilization F
  --mode check|smoke|test-worker|run
  --offline
  --skip-install
  --verbose
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

PROFILE="$1"
shift

ACCELERATOR=""
EXPECTED_DEVICE_COUNT=""
DEVICE_IDS="${MFE_DEVICE_IDS:-}"
MODEL_PATH="${MFE_MODEL_PATH:-}"
DATA_DIR="${MFE_DATA_DIR:-}"
OUTPUT_DIR="${MFE_OUTPUT_DIR:-}"
DATASET="gsm8k"
YAML_FILE="adv_reason_3.yaml"
NUM="20"
TEMPLATES_DIR="templates"
MAX_MODEL_LEN="${MFE_MAX_MODEL_LEN:-}"
GPU_MEMORY_UTILIZATION="${MFE_GPU_MEMORY_UTILIZATION:-}"
MODE="run"
OFFLINE="${MFE_OFFLINE:-0}"
INSTALL="1"
VERBOSE="0"

case "$PROFILE" in
  lab-a800)
    ACCELERATOR="cuda"
    ;;
  company-ascend)
    ACCELERATOR="ascend"
    ;;
  custom)
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    usage
    exit 2
    ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator)
      ACCELERATOR="$2"
      shift 2
      ;;
    --expected-device-count)
      EXPECTED_DEVICE_COUNT="$2"
      shift 2
      ;;
    --device-ids)
      DEVICE_IDS="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --yaml)
      YAML_FILE="$2"
      shift 2
      ;;
    --num)
      NUM="$2"
      shift 2
      ;;
    --templates-dir)
      TEMPLATES_DIR="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --gpu-memory-utilization)
      GPU_MEMORY_UTILIZATION="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --offline)
      OFFLINE="1"
      shift
      ;;
    --skip-install)
      INSTALL="0"
      shift
      ;;
    --verbose)
      VERBOSE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$ACCELERATOR" ]]; then
  echo "--accelerator is required for custom profile" >&2
  exit 2
fi

case "$MODE" in
  check|smoke|test-worker|run) ;;
  *)
    echo "--mode must be one of: check, smoke, test-worker, run" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

export MFE_ACCELERATOR="$ACCELERATOR"
if [[ -n "$DEVICE_IDS" ]]; then
  export MFE_DEVICE_IDS="$DEVICE_IDS"
fi
if [[ -n "$MODEL_PATH" ]]; then
  export MFE_MODEL_PATH="$MODEL_PATH"
fi
if [[ -n "$DATA_DIR" ]]; then
  export MFE_DATA_DIR="$DATA_DIR"
else
  export MFE_DATA_DIR="$PROJECT_ROOT/data"
  DATA_DIR="$MFE_DATA_DIR"
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  export MFE_OUTPUT_DIR="$OUTPUT_DIR"
fi
if [[ -n "$MAX_MODEL_LEN" ]]; then
  export MFE_MAX_MODEL_LEN="$MAX_MODEL_LEN"
fi
if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
  export MFE_GPU_MEMORY_UTILIZATION="$GPU_MEMORY_UTILIZATION"
fi
if [[ "$OFFLINE" == "1" || "$OFFLINE" == "true" ]]; then
  export MFE_OFFLINE=1
fi

if [[ "$INSTALL" == "1" ]]; then
  python -m pip install -e . --no-deps
fi

if [[ "$MODE" != "test-worker" ]]; then
  CHECK_ARGS=(--accelerator "$ACCELERATOR" --data-dir "$DATA_DIR" --require-data-dir --require-gsm8k --require-homogeneous)
  if [[ -n "$EXPECTED_DEVICE_COUNT" ]]; then
    CHECK_ARGS+=(--expected-device-count "$EXPECTED_DEVICE_COUNT")
  fi
  if [[ -n "$DEVICE_IDS" ]]; then
    CHECK_ARGS+=(--device-ids "$DEVICE_IDS")
  fi
  if [[ -n "$MODEL_PATH" ]]; then
    CHECK_ARGS+=(--model-path "$MODEL_PATH")
    CHECK_ARGS+=(--require-model-path)
  fi
  if [[ -z "$MODEL_PATH" && ( "$MODE" == "smoke" || "$MODE" == "run" ) ]]; then
    CHECK_ARGS+=(--require-model-path)
  fi
  if [[ "$OFFLINE" == "1" || "$OFFLINE" == "true" ]]; then
    CHECK_ARGS+=(--offline)
  fi
  python -m mfe.scripts.check_runtime_env "${CHECK_ARGS[@]}"
fi

if [[ "$MODE" == "check" ]]; then
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_ARGS=(--accelerator "$ACCELERATOR" --model-path "$MODEL_PATH")
  if [[ -n "$DEVICE_IDS" ]]; then
    SMOKE_ARGS+=(--device-ids "$DEVICE_IDS")
  fi
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    SMOKE_ARGS+=(--max-model-len "$MAX_MODEL_LEN")
  fi
  if [[ -n "$GPU_MEMORY_UTILIZATION" ]]; then
    SMOKE_ARGS+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
  fi
  if [[ "$OFFLINE" == "1" || "$OFFLINE" == "true" ]]; then
    SMOKE_ARGS+=(--offline)
  fi
  python -m mfe.scripts.smoke_vllm "${SMOKE_ARGS[@]}"
  exit 0
fi

CLIENT_ARGS=(--dataset "$DATASET" --yaml "$YAML_FILE" -n "$NUM" --templates-dir "$TEMPLATES_DIR" --data-dir "$DATA_DIR" --accelerator "$ACCELERATOR")
if [[ -n "$MODEL_PATH" ]]; then
  CLIENT_ARGS+=(--model-path "$MODEL_PATH")
fi
if [[ -n "$OUTPUT_DIR" ]]; then
  CLIENT_ARGS+=(--output-dir "$OUTPUT_DIR")
fi
if [[ -n "$DEVICE_IDS" ]]; then
  CLIENT_ARGS+=(--device-ids "$DEVICE_IDS")
fi
if [[ "$OFFLINE" == "1" || "$OFFLINE" == "true" ]]; then
  CLIENT_ARGS+=(--offline)
fi
if [[ "$VERBOSE" == "1" ]]; then
  CLIENT_ARGS+=(-v)
fi
if [[ "$MODE" == "test-worker" ]]; then
  CLIENT_ARGS+=(--test-worker --worker-delay 0)
fi

python -m mfe.scripts.client "${CLIENT_ARGS[@]}"
