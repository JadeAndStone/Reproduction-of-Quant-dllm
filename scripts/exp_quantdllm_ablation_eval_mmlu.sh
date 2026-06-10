#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
MODEL_ROOT="${MODEL_ROOT:-/root/data-fs/Quant-dllm/outputs/ablations}"
RESULT_ROOT="${RESULT_ROOT:-/root/data-fs/Quant-dllm/benchmark_results/ablations_mmlu}"
HF_HOME="${HF_HOME:-/root/data-tmp/hf_eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

TASKS="${TASKS:-mmlu}"
MC_NUM="${MC_NUM:-128}"
MMLU_MC_NUM="${MMLU_MC_NUM:-$MC_NUM}"
LL_BATCH_SIZE="${LL_BATCH_SIZE:-1}"
USE_OFFLOAD="${USE_OFFLOAD:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-12GB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-60GB}"
DTYPE="${DTYPE:-float16}"
LIMIT="${LIMIT:-}"
OFFLINE="${OFFLINE:-1}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_LIST_FILE="${MODEL_LIST_FILE:-}"
MODEL_GLOB="${MODEL_GLOB:-llada8b_*}"

mkdir -p "$RESULT_ROOT"

run_eval() {
  local run_id="$1"
  local model_path="$2"
  local output_dir="$RESULT_ROOT/$run_id"
  local log_file="$RESULT_ROOT/${run_id}.log"

  if [ ! -d "$model_path" ]; then
    echo "Skip missing model: $model_path" >&2
    return 0
  fi

  local cmd=(
    bash "$ROOT/llada_benchmark.sh"
  )

  echo "===== MMLU eval: $run_id ====="
  echo "model_path=$model_path"
  echo "output_dir=$output_dir"
  echo "log_file=$log_file"

  if [ "$DRY_RUN" = "1" ]; then
    RUN_ID="$run_id" MODEL_PATH="$model_path" OUTPUT_DIR="$output_dir" HF_HOME="$HF_HOME" OFFLINE="$OFFLINE" \
      TASKS="$TASKS" MC_NUM="$MC_NUM" MMLU_MC_NUM="$MMLU_MC_NUM" LL_BATCH_SIZE="$LL_BATCH_SIZE" \
      USE_OFFLOAD="$USE_OFFLOAD" MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
      DTYPE="$DTYPE" PYTHON_BIN="$PYTHON_BIN" LIMIT="$LIMIT" DRY_RUN=1 "${cmd[@]}"
    return 0
  fi

  RUN_ID="$run_id" MODEL_PATH="$model_path" OUTPUT_DIR="$output_dir" HF_HOME="$HF_HOME" OFFLINE="$OFFLINE" \
    TASKS="$TASKS" MC_NUM="$MC_NUM" MMLU_MC_NUM="$MMLU_MC_NUM" LL_BATCH_SIZE="$LL_BATCH_SIZE" \
    USE_OFFLOAD="$USE_OFFLOAD" MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
    DTYPE="$DTYPE" PYTHON_BIN="$PYTHON_BIN" LIMIT="$LIMIT" "${cmd[@]}" > "$log_file" 2>&1
}

if [ -n "$MODEL_LIST_FILE" ]; then
  matched=0
  while IFS=$'\t ' read -r run_id model_path; do
    if [ -z "${run_id:-}" ] || [[ "$run_id" == \#* ]]; then
      continue
    fi
    matched=$((matched + 1))
    run_eval "$run_id" "$model_path"
  done < "$MODEL_LIST_FILE"
else
  matched=0
  while IFS= read -r model_path; do
    matched=$((matched + 1))
    run_id="$(basename "$model_path")"
    run_eval "$run_id" "$model_path"
  done < <(find "$MODEL_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$MODEL_GLOB" | sort)
fi

if [ "$matched" -eq 0 ]; then
  echo "No models matched. MODEL_ROOT=$MODEL_ROOT MODEL_GLOB=$MODEL_GLOB MODEL_LIST_FILE=${MODEL_LIST_FILE:-<unset>}" >&2
fi
