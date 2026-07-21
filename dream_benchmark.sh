#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_HOME="${HF_HOME:-/root/data-tmp/hf_eval}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

OFFLINE="${OFFLINE:-1}"
if [ "$OFFLINE" = "1" ]; then
  export HF_DATASETS_OFFLINE=1
  export HF_HUB_OFFLINE=1
fi

HARNESS_DIR="${HARNESS_DIR:-/root/data-fs/QDLM/lm-evaluation-harness}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Dream snapshot.}"
RUN_ID="${RUN_ID:-$(basename "$MODEL_PATH" | tr -cs '[:alnum:]_.-' '_')}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/benchmark_results/$RUN_ID}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-/root/data-tmp/offload/$RUN_ID}"

TASKS="${TASKS:-piqa,arc_easy,arc_challenge,winogrande,bbh_zeroshot}"
LIMIT="${LIMIT:-}"
MC_NUM="${MC_NUM:-128}"
LL_BATCH_SIZE="${LL_BATCH_SIZE:-1}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-float16}"
CLASSIFIER_FREE_GUIDANCE="${CLASSIFIER_FREE_GUIDANCE:-1.0}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_MEMORY_PER_GPU="${MAX_MEMORY_PER_GPU:-13GB}"
MAX_CPU_MEMORY="${MAX_CPU_MEMORY:-80GB}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-0}"
FEWSHOT_AS_MULTITURN="${FEWSHOT_AS_MULTITURN:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUTPUT_DIR" "$OFFLOAD_FOLDER" \
  "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE"
cd "$HARNESS_DIR" || exit 1

FAILED_TASKS=()

run_lm_eval() {
  local task="$1"
  local fewshot="$2"
  local batch_size="$3"
  local output_name="$4"
  local task_offload="$OFFLOAD_FOLDER/$output_name"
  mkdir -p "$task_offload"

  local model_args="pretrained=$MODEL_PATH,dtype=$DTYPE,nll_type=mc,log_type=ftb,mc_num=$MC_NUM,classifier_free_guidance=$CLASSIFIER_FREE_GUIDANCE,diffusion_steps=$DIFFUSION_STEPS,max_new_tokens=$MAX_NEW_TOKENS,device_map=auto,max_memory_per_gpu=$MAX_MEMORY_PER_GPU,max_cpu_memory=$MAX_CPU_MEMORY,offload_folder=$task_offload"
  local cmd=(
    "$PYTHON_BIN" -m lm_eval
    --model dream_base
    --model_args "$model_args"
    --tasks "$task"
    --num_fewshot "$fewshot"
    --batch_size "$batch_size"
    --output_path "$OUTPUT_DIR/$output_name.json"
  )

  if [ -n "$LIMIT" ]; then
    cmd+=(--limit "$LIMIT")
  fi
  if [ "$APPLY_CHAT_TEMPLATE" = "1" ]; then
    cmd+=(--apply_chat_template)
  fi
  if [ "$FEWSHOT_AS_MULTITURN" = "1" ]; then
    if [ "$APPLY_CHAT_TEMPLATE" != "1" ]; then
      echo "FEWSHOT_AS_MULTITURN=1 requires APPLY_CHAT_TEMPLATE=1" >&2
      FAILED_TASKS+=("$task")
      return
    fi
    cmd+=(--fewshot_as_multiturn)
  fi

  echo "===== Running $task ====="
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return
  fi
  if "${cmd[@]}"; then
    echo "===== Finished $task ====="
  else
    echo "===== Failed $task ====="
    FAILED_TASKS+=("$task")
  fi
}

cat <<EOF
===== Dream benchmark config =====
RUN_ID=$RUN_ID
MODEL_PATH=$MODEL_PATH
OUTPUT_DIR=$OUTPUT_DIR
TASKS=$TASKS
LIMIT=${LIMIT:-<full>}
MC_NUM=$MC_NUM
CLASSIFIER_FREE_GUIDANCE=$CLASSIFIER_FREE_GUIDANCE
DIFFUSION_STEPS=$DIFFUSION_STEPS
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
DTYPE=$DTYPE
MAX_MEMORY_PER_GPU=$MAX_MEMORY_PER_GPU
MAX_CPU_MEMORY=$MAX_CPU_MEMORY
APPLY_CHAT_TEMPLATE=$APPLY_CHAT_TEMPLATE
FEWSHOT_AS_MULTITURN=$FEWSHOT_AS_MULTITURN
OFFLINE=$OFFLINE
==================================
EOF

SELECTED_TASKS=($(echo "$TASKS" | tr ',' ' '))
for task in "${SELECTED_TASKS[@]}"; do
  case "$task" in
    piqa|arc_easy|arc_challenge|hellaswag)
      run_lm_eval "$task" 0 "$LL_BATCH_SIZE" "$task"
      ;;
    mmlu)
      run_lm_eval mmlu 5 "$LL_BATCH_SIZE" mmlu_5shot
      ;;
    winogrande)
      run_lm_eval winogrande 5 "$LL_BATCH_SIZE" winogrande_5shot
      ;;
    bbh|bbh_zeroshot)
      run_lm_eval bbh_zeroshot 0 "$GEN_BATCH_SIZE" bbh_zeroshot
      ;;
    *)
      echo "Unknown task: $task" >&2
      FAILED_TASKS+=("$task")
      ;;
  esac
done

if [ "${#FAILED_TASKS[@]}" -ne 0 ]; then
  echo "Failed tasks: ${FAILED_TASKS[*]}"
  exit 1
fi
