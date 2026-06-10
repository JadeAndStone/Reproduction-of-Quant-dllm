#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_NAME="${MODEL_NAME:-dream7b_base}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the local model directory.}"
C4_SOURCE="${C4_SOURCE:-https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data-fs/Quant-dllm/outputs/cross_model}"
Z_ROOT="${Z_ROOT:-/root/data-tmp/quant_dllm_z_cache_cross_model}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/root/data-tmp/offload/quant_dllm_cross_model}"

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
ALLOCATE_RATIO="${ALLOCATE_RATIO:-0.05}"
GPU_MEMORY="${GPU_MEMORY:-13GiB}"
CPU_MEMORY="${CPU_MEMORY:-80GiB}"
DRY_RUN="${DRY_RUN:-0}"

DAQ_GRANULARITY="${DAQ_GRANULARITY:-tile}"
ABMP_GRANULARITY="${ABMP_GRANULARITY:-tile}"
ABMP_RANK_SCOPE="${ABMP_RANK_SCOPE:-transformer_block}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-128}"

run_id="${MODEL_NAME}_abmp05_tile128_n${NSAMPLES}"
output_dir="$OUTPUT_ROOT/$run_id"
z_cache_dir="$Z_ROOT/${MODEL_NAME}_c4_n${NSAMPLES}_s${SEQLEN}_t${NUM_STEPS}_g${GROUP_SIZE}"
offload_folder="$OFFLOAD_ROOT/$run_id"

cmd=(
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
  --allocate_ratio "$ALLOCATE_RATIO"
  --damping "$DAMPING"
  --daq_steps "$DAQ_STEPS"
  --imp_lamda "$IMP_LAMDA"
  --reuse_z
  --z_cache_dir "$z_cache_dir"
  --gpu_memory "$GPU_MEMORY"
  --cpu_memory "$CPU_MEMORY"
  --offload_folder "$offload_folder"
  --daq_granularity "$DAQ_GRANULARITY"
  --abmp_granularity "$ABMP_GRANULARITY"
  --abmp_rank_scope "$ABMP_RANK_SCOPE"
  --quant_block_size "$QUANT_BLOCK_SIZE"
  --daq_device cpu
)

echo "===== Cross-model Quant-DLLM template ====="
echo "MODEL_NAME=$MODEL_NAME"
echo "MODEL_PATH=$MODEL_PATH"
echo "output_dir=$output_dir"
echo "z_cache_dir=$z_cache_dir"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [ "$DRY_RUN" != "1" ]; then
  "${cmd[@]}"
fi
