#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}"
C4_SOURCE="${C4_SOURCE:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}"

Z_CACHE_DIR="${Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/z_cache_no_full_visible_centered_z_tile128}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/data-fs/Quant-dllm/outputs/row_center_no_full_visible_centered_z/llada8b_daq_rsr_w_dor_row_center_centered_z_tile128_n128}"
OFFLOAD_FOLDER="${OFFLOAD_FOLDER:-/root/data-tmp/offload/row_center_no_full_visible_centered_z/llada8b_daq_rsr_w_dor_row_center_centered_z_tile128_n128}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-4096}"
SEED="${SEED:-42}"
NUM_STEPS="${NUM_STEPS:-20}"
GAMMA="${GAMMA:-0.25}"
SCHEDULE="${SCHEDULE:-linear}"
GROUP_SIZE="${GROUP_SIZE:-128}"
DAMPING="${DAMPING:-0.01}"

DAQ_STEPS="${DAQ_STEPS:-10}"
IMP_LAMDA="${IMP_LAMDA:-2.0}"
DAQ_COMPONENT="${DAQ_COMPONENT:-rsr_w_dor}"
DEFAULT_BITS="${DEFAULT_BITS:-2}"
SAVE_DTYPE="${SAVE_DTYPE:-fp16}"
Z_SAVE_DTYPE="${Z_SAVE_DTYPE:-fp32}"

DAQ_GRANULARITY="${DAQ_GRANULARITY:-tile}"
ABMP_GRANULARITY="${ABMP_GRANULARITY:-tile}"
ABMP_RANK_SCOPE="${ABMP_RANK_SCOPE:-transformer_block}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-128}"
DAQ_DEVICE="${DAQ_DEVICE:-cpu}"
LOG_UNIT_INTERVAL="${LOG_UNIT_INTERVAL:-0}"

GPU_MEMORY="${GPU_MEMORY:-15GiB}"
CPU_MEMORY="${CPU_MEMORY:-80GiB}"
START_BLOCK="${START_BLOCK:-}"
END_BLOCK="${END_BLOCK:-}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$(dirname "$OUTPUT_DIR")" "$Z_CACHE_DIR" "$OFFLOAD_FOLDER"

cmd=(
  "$PYTHON_BIN" "$ROOT/main.py"
  --model_path "$MODEL_PATH"
  --output_dir "$OUTPUT_DIR"
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
  --offload_folder "$OFFLOAD_FOLDER"
  --daq_granularity "$DAQ_GRANULARITY"
  --abmp_granularity "$ABMP_GRANULARITY"
  --abmp_rank_scope "$ABMP_RANK_SCOPE"
  --quant_block_size "$QUANT_BLOCK_SIZE"
  --daq_device "$DAQ_DEVICE"
  --log_unit_interval "$LOG_UNIT_INTERVAL"
  --daq_row_center
  --z_row_center
  --mcs_no_full_visible
)

if [ -n "$START_BLOCK" ]; then
  cmd+=(--start_block "$START_BLOCK")
fi
if [ -n "$END_BLOCK" ]; then
  cmd+=(--end_block "$END_BLOCK")
fi

cat <<EOM
===== Quant row-center with no-full-visible centered-Z =====
ROOT=$ROOT
PYTHON_BIN=$PYTHON_BIN
MODEL_PATH=$MODEL_PATH
OUTPUT_DIR=$OUTPUT_DIR
Z_CACHE_DIR=$Z_CACHE_DIR
C4_SOURCE=$C4_SOURCE
DAQ_COMPONENT=$DAQ_COMPONENT
DAQ_ROW_CENTER=1
Z_ROW_CENTER=1
MCS_NO_FULL_VISIBLE=1
DAQ_GRANULARITY=$DAQ_GRANULARITY
ABMP_GRANULARITY=$ABMP_GRANULARITY
ABMP_RANK_SCOPE=$ABMP_RANK_SCOPE
QUANT_BLOCK_SIZE=$QUANT_BLOCK_SIZE
DAQ_DEVICE=$DAQ_DEVICE
===============================================
EOM

printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

exec "${cmd[@]}"
