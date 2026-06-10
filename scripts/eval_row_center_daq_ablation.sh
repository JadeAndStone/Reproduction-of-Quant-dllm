#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-/root/data-fs/Quant-dllm/outputs/row_center_daq_ablation}"
RESULT_ROOT="${RESULT_ROOT:-/root/data-fs/Quant-dllm/benchmark_results}"

DAQ_COMPONENT="${DAQ_COMPONENT:-rsr_w_dor}"
NSAMPLES="${NSAMPLES:-128}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-128}"
VARIANTS="${VARIANTS:-no_row_center,row_center}"

TASKS="${TASKS:-piqa,arc_challenge,gsm8k}"
LIMIT="${LIMIT:-200}"
MC_NUM="${MC_NUM:-128}"
MMLU_MC_NUM="${MMLU_MC_NUM:-$MC_NUM}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
LL_BATCH_SIZE="${LL_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
USE_OFFLOAD="${USE_OFFLOAD:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-15GB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-0GB}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
HF_HOME="${HF_HOME:-/root/data-tmp/hf_eval}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
OFFLINE="${OFFLINE:-1}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-1}"
HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

cd "$ROOT"

TASK_TAG="${TASKS//,/+}"
TASK_TAG="${TASK_TAG// /_}"

IFS=',' read -ra VARIANT_LIST <<< "$VARIANTS"
for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant// /}"
  model_name="llada8b_daq_${DAQ_COMPONENT}_${variant}_tile${QUANT_BLOCK_SIZE}_n${NSAMPLES}"
  model_path="$MODEL_ROOT/$model_name"
  if [ ! -d "$model_path" ]; then
    echo "Missing model directory: $model_path" >&2
    exit 1
  fi

  run_id="eval_${model_name}_${TASK_TAG}_limit${LIMIT}"
  log="$RESULT_ROOT/${run_id}_$(date +%Y%m%d_%H%M%S).log"
  output_dir="$RESULT_ROOT/$run_id"
  offload_folder="/root/data-tmp/offload/$run_id"

  echo "===== Eval row-centering ablation: $model_name ====="
  echo "model_path=$model_path"
  echo "tasks=$TASKS"
  echo "limit=$LIMIT"
  echo "log=$log"

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_ID="$run_id" \
  MODEL_PATH="$model_path" \
  OUTPUT_DIR="$output_dir" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
  HF_MODULES_CACHE="$HF_MODULES_CACHE" \
  HF_ALLOW_CODE_EVAL="$HF_ALLOW_CODE_EVAL" \
  CONFIRM_RUN_UNSAFE_CODE="$CONFIRM_RUN_UNSAFE_CODE" \
  OFFLINE="$OFFLINE" \
  TASKS="$TASKS" \
  LIMIT="$LIMIT" \
  MC_NUM="$MC_NUM" \
  MMLU_MC_NUM="$MMLU_MC_NUM" \
  GEN_BATCH_SIZE="$GEN_BATCH_SIZE" \
  LL_BATCH_SIZE="$LL_BATCH_SIZE" \
  DTYPE="$DTYPE" \
  USE_OFFLOAD="$USE_OFFLOAD" \
  MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" \
  MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
  OFFLOAD_FOLDER="$offload_folder" \
  bash ./llada_benchmark.sh > "$log" 2>&1

  echo "finished $model_name, log=$log"
done
