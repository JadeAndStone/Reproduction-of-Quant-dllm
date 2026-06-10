#!/usr/bin/env bash
set -uo pipefail

export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="${HF_HOME:-/root/data-fs/hf_shared}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

OFFLINE="${OFFLINE:-0}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-0}"
if [ "$OFFLINE" = "1" ]; then
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

QDLM_ROOT="${QDLM_ROOT:-/root/data-fs/QDLM}"
HARNESS_DIR="${HARNESS_DIR:-$QDLM_ROOT/lm-evaluation-harness}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${MODEL_PATH:-/root/data-fs/Quant-dllm/outputs/llada_daq_fake_quant}"
RUN_ID="${RUN_ID:-$(basename "$MODEL_PATH" | tr -cs '[:alnum:]_.-' '_')}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data-fs/Quant-dllm/benchmark_results/$RUN_ID}"

MC_NUM="${MC_NUM:-128}"
LL_BATCH_SIZE="${LL_BATCH_SIZE:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
MMLU_MC_NUM="${MMLU_MC_NUM:-$MC_NUM}"
DTYPE="${DTYPE:-float16}"

USE_OFFLOAD="${USE_OFFLOAD:-1}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-12GB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-60GB}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-/root/data-tmp/offload/$RUN_ID}"

TASK_GROUP="${TASK_GROUP:-all}"
TASKS="${TASKS:-}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE"
if [ "$USE_OFFLOAD" = "1" ]; then
  mkdir -p "$OFFLOAD_FOLDER"
fi

cd "$HARNESS_DIR" || exit 1

FAILED_TASKS=()

build_model_args() {
  local mc_num="$1"
  local steps="$2"
  local gen_length="$3"
  local block_length="$4"
  local args="model_path=$MODEL_PATH,dtype=$DTYPE,mc_num=$mc_num,is_check_greedy=False,steps=$steps,gen_length=$gen_length,block_length=$block_length"
  if [ "$USE_OFFLOAD" = "1" ]; then
    args="$args,device_map=auto,max_memory_per_gpu=$MAX_MEMORY_PER_GPU,max_cpu_memory=$MAX_CPU_MEMORY,offload_folder=$OFFLOAD_FOLDER"
  fi
  echo "$args"
}

print_config() {
  cat <<EOF
===== LLaDA benchmark config =====
RUN_ID=$RUN_ID
MODEL_PATH=$MODEL_PATH
OUTPUT_DIR=$OUTPUT_DIR
TASK_GROUP=$TASK_GROUP
TASKS=${TASKS:-<from group>}
MC_NUM=$MC_NUM
MMLU_MC_NUM=$MMLU_MC_NUM
DTYPE=$DTYPE
USE_OFFLOAD=$USE_OFFLOAD
OFFLOAD_FOLDER=$OFFLOAD_FOLDER
HF_HOME=$HF_HOME
OFFLINE=$OFFLINE
CONFIRM_RUN_UNSAFE_CODE=$CONFIRM_RUN_UNSAFE_CODE
PYTHON_BIN=$PYTHON_BIN
==================================
EOF
}

resolve_tasks() {
  if [ -n "$TASKS" ]; then
    echo "$TASKS" | tr ',' ' '
    return
  fi

  case "$TASK_GROUP" in
    paper_mc)
      echo "piqa arc_easy arc_challenge hellaswag mmlu winogrande bbh_zeroshot"
      ;;
    reasoning)
      echo "gsm8k minerva_math"
      ;;
    code)
      echo "humaneval mbpp"
      ;;
    all)
      echo "piqa arc_easy arc_challenge hellaswag mmlu winogrande bbh_zeroshot gsm8k minerva_math humaneval mbpp"
      ;;
    *)
      echo "Unknown TASK_GROUP=$TASK_GROUP. Use paper_mc, reasoning, code, all, or set TASKS." >&2
      exit 2
      ;;
  esac
}

run_lm_eval() {
  local task="$1"
  local fewshot="$2"
  local mc_num="$3"
  local batch_size="$4"
  local steps="$5"
  local gen_length="$6"
  local block_length="$7"
  local output_name="$8"
  local model_args
  model_args="$(build_model_args "$mc_num" "$steps" "$gen_length" "$block_length")"

  local cmd=(
    "$PYTHON_BIN" -m lm_eval
    --model llada_dist
    --model_args "$model_args"
    --tasks "$task"
    --num_fewshot "$fewshot"
    --batch_size "$batch_size"
    --output_path "$OUTPUT_DIR/$output_name.json"
  )

  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi

  if [ "$CONFIRM_RUN_UNSAFE_CODE" = "1" ]; then
    cmd+=(--confirm_run_unsafe_code)
  fi

  echo "===== Running ${task} ====="
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [ "$DRY_RUN" = "1" ]; then
    echo "===== Dry-run skipped ${task} ====="
    return 0
  fi

  if "${cmd[@]}"; then
    echo "===== Finished ${task} ====="
  else
    echo "===== Failed ${task} ====="
    FAILED_TASKS+=("$task")
  fi
}

run_task() {
  local task="$1"

  case "$task" in
    piqa)
      run_lm_eval piqa 0 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 piqa
      ;;
    arc_easy)
      run_lm_eval arc_easy 0 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 arc_easy
      ;;
    arc_challenge)
      run_lm_eval arc_challenge 0 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 arc_challenge
      ;;
    hellaswag)
      run_lm_eval hellaswag 0 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 hellaswag
      ;;
    mmlu)
      run_lm_eval mmlu 5 "$MMLU_MC_NUM" "$LL_BATCH_SIZE" 256 256 32 mmlu_5shot
      ;;
    winogrande)
      run_lm_eval winogrande 5 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 winogrande_5shot
      ;;
    bbh|bbh_zeroshot)
      run_lm_eval bbh_zeroshot 0 "$MC_NUM" "$LL_BATCH_SIZE" 256 256 32 bbh_zeroshot
      ;;
    gsm8k)
      run_lm_eval gsm8k 4 "$MC_NUM" "$GEN_BATCH_SIZE" 256 256 32 gsm8k
      ;;
    minerva_math)
      run_lm_eval minerva_math 0 "$MC_NUM" "$GEN_BATCH_SIZE" 256 256 64 minerva_math
      ;;
    humaneval)
      run_lm_eval humaneval 0 "$MC_NUM" "$GEN_BATCH_SIZE" 512 512 32 humaneval
      ;;
    mbpp)
      run_lm_eval mbpp 3 "$MC_NUM" "$GEN_BATCH_SIZE" 512 512 32 mbpp
      ;;
    *)
      echo "Unknown task: $task" >&2
      FAILED_TASKS+=("$task")
      ;;
  esac
}

print_config
SELECTED_TASKS=($(resolve_tasks))
echo "Selected tasks: ${SELECTED_TASKS[*]}"

for task in "${SELECTED_TASKS[@]}"; do
  run_task "$task"
done

if [ "${#FAILED_TASKS[@]}" -ne 0 ]; then
  echo "Failed tasks: ${FAILED_TASKS[*]}"
  exit 1
fi
