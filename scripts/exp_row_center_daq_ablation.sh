#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}"
C4_SOURCE="${C4_SOURCE:-https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data-fs/Quant-dllm/outputs/row_center_daq_ablation}"
Z_CACHE_DIR="${Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/llada_all2_group128_daq_newz/z_cache}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/root/data-tmp/offload/row_center_daq_ablation}"

RUN_SET="${RUN_SET:-both}"
DRY_RUN="${DRY_RUN:-0}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-4096}"
SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-20}"
GAMMA="${GAMMA:-0.25}"
SCHEDULE="${SCHEDULE:-linear}"
MAX_DOCS="${MAX_DOCS:-}"
MAX_MCS_SAMPLES="${MAX_MCS_SAMPLES:-}"

GROUP_SIZE="${GROUP_SIZE:-128}"
DAMPING="${DAMPING:-0.01}"
DAQ_STEPS="${DAQ_STEPS:-10}"
IMP_LAMDA="${IMP_LAMDA:-2.0}"
DAQ_COMPONENT="${DAQ_COMPONENT:-rsr_w_dor}"
DEFAULT_BITS="${DEFAULT_BITS:-2}"
SAVE_DTYPE="${SAVE_DTYPE:-fp16}"
Z_SAVE_DTYPE="${Z_SAVE_DTYPE:-fp32}"

DAQ_GRANULARITY="${DAQ_GRANULARITY:-tile}"
ABMP_GRANULARITY="${ABMP_GRANULARITY:-$DAQ_GRANULARITY}"
ABMP_RANK_SCOPE="${ABMP_RANK_SCOPE:-transformer_block}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-128}"
DAQ_DEVICE="${DAQ_DEVICE:-cpu}"

GPU_MEMORY="${GPU_MEMORY:-15GiB}"
CPU_MEMORY="${CPU_MEMORY:-80GiB}"
START_BLOCK="${START_BLOCK:-}"
END_BLOCK="${END_BLOCK:-}"
LOG_UNIT_INTERVAL="${LOG_UNIT_INTERVAL:-0}"

mkdir -p "$OUTPUT_ROOT" "$OFFLOAD_ROOT"

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

run_quant() {
  local variant="$1"
  local row_center="$2"
  local run_name="llada8b_daq_${DAQ_COMPONENT}_${variant}_tile${QUANT_BLOCK_SIZE}_n${NSAMPLES}"
  local output_dir="$OUTPUT_ROOT/$run_name"
  local offload_folder="$OFFLOAD_ROOT/$run_name"

  local cmd=(
    "$PYTHON_BIN" "$ROOT/main.py"
    --model_path "$MODEL_PATH"
    --output_dir "$output_dir"
    --source "$C4_SOURCE"
    --nsamples "$NSAMPLES"
    --seed "$SEED"
    --seqlen "$SEQLEN"
    --num-steps "$NUM_STEPS"
    --gamma "$GAMMA"
    --group_size "$GROUP_SIZE"
    --schedule "$SCHEDULE"
    --allocate_ratio "0.05"
    --damping "$DAMPING"
    --daq_steps "$DAQ_STEPS"
    --imp_lamda "$IMP_LAMDA"
    --daq_component "$DAQ_COMPONENT"
    --default_bits "$DEFAULT_BITS"
    --reuse_z
    --z_cache_dir "$Z_CACHE_DIR"
    --z_save_dtype "$Z_SAVE_DTYPE"
    --save_dtype "$SAVE_DTYPE"
    --gpu_memory "$GPU_MEMORY"
    --cpu_memory "$CPU_MEMORY"
    --offload_folder "$offload_folder"
    --daq_granularity "$DAQ_GRANULARITY"
    --abmp_granularity "$ABMP_GRANULARITY"
    --abmp_rank_scope "$ABMP_RANK_SCOPE"
    --quant_block_size "$QUANT_BLOCK_SIZE"
    --daq_device "$DAQ_DEVICE"
    --log_unit_interval "$LOG_UNIT_INTERVAL"
  )

  if [ -n "$MAX_DOCS" ]; then
    cmd+=(--max-docs "$MAX_DOCS")
  fi
  if [ -n "$MAX_MCS_SAMPLES" ]; then
    cmd+=(--max_mcs_samples "$MAX_MCS_SAMPLES")
  fi
  if [ -n "$START_BLOCK" ]; then
    cmd+=(--start_block "$START_BLOCK")
  fi
  if [ -n "$END_BLOCK" ]; then
    cmd+=(--end_block "$END_BLOCK")
  fi
  if [ "$row_center" = "1" ]; then
    cmd+=(--daq_row_center)
  fi

  echo "===== Row-centering DAQ ablation: $run_name ====="
  echo "output_dir=$output_dir"
  echo "z_cache_dir=$Z_CACHE_DIR"
  echo "daq_component=$DAQ_COMPONENT"
  echo "daq_row_center=$row_center"
  echo "abmp_rank_scope=$ABMP_RANK_SCOPE"
  run_cmd "${cmd[@]}"
}

case "$RUN_SET" in
  both)
    run_quant "no_row_center" "0"
    run_quant "row_center" "1"
    ;;
  no_row_center)
    run_quant "no_row_center" "0"
    ;;
  row_center)
    run_quant "row_center" "1"
    ;;
  *)
    echo "Unknown RUN_SET=$RUN_SET. Use both, no_row_center, or row_center." >&2
    exit 2
    ;;
esac
