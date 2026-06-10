#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QDLM_ROOT="${QDLM_ROOT:-/root/data-fs/QDLM}"

MODEL_PATH="${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}"
C4_SOURCE="${C4_SOURCE:-https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz}"
HF_HOME="${HF_HOME:-/root/data-tmp/hf_eval}"

WBITS="${WBITS:-2}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DESC_ACT="${DESC_ACT:-0}"
CALIB_NSAMPLES="${CALIB_NSAMPLES:-128}"
CALIB_SEED="${CALIB_SEED:-42}"
CALIB_SEQLEN="${CALIB_SEQLEN:-4096}"
CALIB_MAX_DOCS="${CALIB_MAX_DOCS:-}"

QUANTIZED_MODEL_DIR="${QUANTIZED_MODEL_DIR:-/root/data-fs/Quant-dllm/outputs/gptq/llada8b_gptq_w${WBITS}_g${GROUP_SIZE}_c4n${CALIB_NSAMPLES}}"
RESULTS_DIR="${RESULTS_DIR:-/root/data-fs/Quant-dllm/benchmark_results/gptq_llada8b_w${WBITS}_g${GROUP_SIZE}_c4n${CALIB_NSAMPLES}}"

TASK_GROUP="${TASK_GROUP:-paper_mc}"
TASKS="${TASKS:-}"
MC_NUM="${MC_NUM:-128}"
MMLU_MC_NUM="${MMLU_MC_NUM:-$MC_NUM}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LIMIT="${LIMIT:--1}"
DTYPE="${DTYPE:-float16}"
DEVICE="${DEVICE:-cuda:0}"
USE_TRITON="${USE_TRITON:-0}"
QUANTIZE_ONLY="${QUANTIZE_ONLY:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

export QDLM_ROOT
export HF_HOME
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="$QDLM_ROOT/AutoGPTQ:$QDLM_ROOT/lm-evaluation-harness:${PYTHONPATH:-}"

mkdir -p "$QUANTIZED_MODEL_DIR" "$RESULTS_DIR"

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

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

run_task() {
  local task="$1"
  local fewshot="$2"
  local mc_num="$3"
  local steps="$4"
  local gen_length="$5"
  local block_length="$6"
  local output_name="$7"

  local cmd=(
    "$PYTHON_BIN" "$ROOT/scripts/run_llada_gptq.py"
    --model_path "$MODEL_PATH"
    --quantized_model_dir "$QUANTIZED_MODEL_DIR"
    --results_dir "$RESULTS_DIR"
    --tasks "$task"
    --num_fewshot "$fewshot"
    --limit "$LIMIT"
    --wbits "$WBITS"
    --group_size "$GROUP_SIZE"
    --calib_source "$C4_SOURCE"
    --calib_nsamples "$CALIB_NSAMPLES"
    --calib_seed "$CALIB_SEED"
    --calib_seqlen "$CALIB_SEQLEN"
    --mc_num "$mc_num"
    --batch_size "$BATCH_SIZE"
    --steps "$steps"
    --gen_length "$gen_length"
    --block_length "$block_length"
    --dtype "$DTYPE"
    --device "$DEVICE"
    --output_name "$output_name"
  )

  if [ -n "$CALIB_MAX_DOCS" ]; then
    cmd+=(--calib_max_docs "$CALIB_MAX_DOCS")
  fi
  if [ "$DESC_ACT" = "1" ]; then
    cmd+=(--desc_act)
  fi
  if [ "$USE_TRITON" = "1" ]; then
    cmd+=(--use_triton)
  fi
  if [ "$QUANTIZE_ONLY" = "1" ]; then
    cmd+=(--quantize_only)
  fi
  if [ "$EVAL_ONLY" = "1" ]; then
    cmd+=(--eval_only)
  fi

  echo "===== GPTQ task: $task ====="
  run_cmd "${cmd[@]}"
}

run_one() {
  case "$1" in
    piqa)
      run_task piqa 0 "$MC_NUM" 256 256 32 piqa
      ;;
    arc_easy)
      run_task arc_easy 0 "$MC_NUM" 256 256 32 arc_easy
      ;;
    arc_challenge)
      run_task arc_challenge 0 "$MC_NUM" 256 256 32 arc_challenge
      ;;
    hellaswag)
      run_task hellaswag 0 "$MC_NUM" 256 256 32 hellaswag
      ;;
    mmlu)
      run_task mmlu 5 "$MMLU_MC_NUM" 256 256 32 mmlu_5shot
      ;;
    winogrande)
      run_task winogrande 5 "$MC_NUM" 256 256 32 winogrande_5shot
      ;;
    bbh|bbh_zeroshot)
      run_task bbh_zeroshot 0 "$MC_NUM" 256 256 32 bbh_zeroshot
      ;;
    gsm8k)
      run_task gsm8k 4 "$MC_NUM" 256 256 32 gsm8k
      ;;
    minerva_math)
      run_task minerva_math 0 "$MC_NUM" 256 256 64 minerva_math
      ;;
    humaneval)
      run_task humaneval 0 "$MC_NUM" 512 512 32 humaneval
      ;;
    mbpp)
      run_task mbpp 3 "$MC_NUM" 512 512 32 mbpp
      ;;
    *)
      echo "Unknown task: $1" >&2
      exit 2
      ;;
  esac
}

echo "===== GPTQ baseline config ====="
echo "MODEL_PATH=$MODEL_PATH"
echo "QUANTIZED_MODEL_DIR=$QUANTIZED_MODEL_DIR"
echo "RESULTS_DIR=$RESULTS_DIR"
echo "WBITS=$WBITS GROUP_SIZE=$GROUP_SIZE CALIB_NSAMPLES=$CALIB_NSAMPLES CALIB_SEQLEN=$CALIB_SEQLEN"
echo "TASK_GROUP=$TASK_GROUP TASKS=${TASKS:-<from group>}"
echo "HF_HOME=$HF_HOME"
echo "================================"

for task in $(resolve_tasks); do
  run_one "$task"
  if [ "$QUANTIZE_ONLY" = "1" ]; then
    break
  fi
done
