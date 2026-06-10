#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/data-fs/Quant-dllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}"
C4_SOURCE="${C4_SOURCE:-https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/root/data-fs/Quant-dllm/outputs/ablations}"
Z_ROOT="${Z_ROOT:-/root/data-tmp/quant_dllm_z_cache}"
Z_CACHE_DIR="${Z_CACHE_DIR:-}"
OFFLOAD_ROOT="${OFFLOAD_ROOT:-/root/data-tmp/offload/quant_dllm_ablation}"

RUN_SET="${RUN_SET:-abmp_ratios}"
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
DAQ_COMPONENT="${DAQ_COMPONENT:-custom}"
DEFAULT_BITS="${DEFAULT_BITS:-2}"
SAVE_DTYPE="${SAVE_DTYPE:-fp16}"
Z_SAVE_DTYPE="${Z_SAVE_DTYPE:-fp32}"

DAQ_GRANULARITY="${DAQ_GRANULARITY:-tile}"
ABMP_GRANULARITY="${ABMP_GRANULARITY:-$DAQ_GRANULARITY}"
ABMP_RANK_SCOPE="${ABMP_RANK_SCOPE:-transformer_block}"
QUANT_BLOCK_SIZE="${QUANT_BLOCK_SIZE:-128}"
DAQ_DEVICE="${DAQ_DEVICE:-cpu}"

GPU_MEMORY="${GPU_MEMORY:-13GiB}"
CPU_MEMORY="${CPU_MEMORY:-80GiB}"
START_BLOCK="${START_BLOCK:-}"
END_BLOCK="${END_BLOCK:-}"
LOG_UNIT_INTERVAL="${LOG_UNIT_INTERVAL:-0}"

mkdir -p "$OUTPUT_ROOT" "$Z_ROOT" "$OFFLOAD_ROOT"

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  "$@"
}

z_cache_name() {
  local nsamples="$1"
  echo "llada8b_c4_n${nsamples}_s${SEQLEN}_t${NUM_STEPS}_g${GROUP_SIZE}_d${DAMPING}_${SCHEDULE}"
}

run_quant() {
  local run_name="$1"
  local nsamples="$2"
  local allocate_ratio="$3"
  local skip_abmp="$4"
  local daq_granularity="$5"
  local abmp_granularity="$6"
  local abmp_rank_scope="$7"
  local daq_component="${8:-$DAQ_COMPONENT}"

  local output_dir="$OUTPUT_ROOT/$run_name"
  local z_cache_dir
  if [ -n "$Z_CACHE_DIR" ]; then
    z_cache_dir="$Z_CACHE_DIR"
  else
    z_cache_dir="$Z_ROOT/$(z_cache_name "$nsamples")"
  fi
  local offload_folder="$OFFLOAD_ROOT/$run_name"

  local cmd=(
    "$PYTHON_BIN" "$ROOT/main.py"
    --model_path "$MODEL_PATH"
    --output_dir "$output_dir"
    --source "$C4_SOURCE"
    --nsamples "$nsamples"
    --seed "$SEED"
    --seqlen "$SEQLEN"
    --num-steps "$NUM_STEPS"
    --gamma "$GAMMA"
    --group_size "$GROUP_SIZE"
    --schedule "$SCHEDULE"
    --allocate_ratio "$allocate_ratio"
    --damping "$DAMPING"
    --daq_steps "$DAQ_STEPS"
    --imp_lamda "$IMP_LAMDA"
    --daq_component "$daq_component"
    --default_bits "$DEFAULT_BITS"
    --reuse_z
    --z_cache_dir "$z_cache_dir"
    --z_save_dtype "$Z_SAVE_DTYPE"
    --save_dtype "$SAVE_DTYPE"
    --gpu_memory "$GPU_MEMORY"
    --cpu_memory "$CPU_MEMORY"
    --offload_folder "$offload_folder"
    --daq_granularity "$daq_granularity"
    --abmp_granularity "$abmp_granularity"
    --abmp_rank_scope "$abmp_rank_scope"
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
  if [ "$skip_abmp" = "1" ]; then
    cmd+=(--skip_abmp)
  fi

  echo "===== Quant-DLLM ablation: $run_name ====="
  echo "output_dir=$output_dir"
  echo "z_cache_dir=$z_cache_dir"
  echo "daq_component=$daq_component"
  run_cmd "${cmd[@]}"
}

run_abmp_ratios() {
  run_quant "llada8b_all2_tile128_noabmp_n${NSAMPLES}" "$NSAMPLES" "0.00" "1" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp00_tile128_n${NSAMPLES}" "$NSAMPLES" "0.00" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp05_tile128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp10_tile128_n${NSAMPLES}" "$NSAMPLES" "0.10" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp15_tile128_n${NSAMPLES}" "$NSAMPLES" "0.15" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
}

run_calib_sizes() {
  run_quant "llada8b_abmp05_tile128_n64" "64" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp05_tile128_n128" "128" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
  run_quant "llada8b_abmp05_tile128_n256" "256" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE"
}

run_granularity() {
  run_quant "llada8b_abmp05_weight_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "weight" "weight" "weight"
  run_quant "llada8b_abmp05_column128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "column" "column" "weight"
  run_quant "llada8b_abmp05_tile128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "tile" "tile" "$ABMP_RANK_SCOPE"
}

run_daq_components() {
  run_quant "llada8b_daq_baseline_tile128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE" "baseline"
  run_quant "llada8b_daq_rsr_no_dor_tile128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE" "rsr_no_dor"
  run_quant "llada8b_daq_rsr_w_dor_tile128_n${NSAMPLES}" "$NSAMPLES" "0.05" "0" "$DAQ_GRANULARITY" "$ABMP_GRANULARITY" "$ABMP_RANK_SCOPE" "rsr_w_dor"
}

case "$RUN_SET" in
  abmp_ratios)
    run_abmp_ratios
    ;;
  calib_sizes)
    run_calib_sizes
    ;;
  granularity)
    run_granularity
    ;;
  daq_components)
    run_daq_components
    ;;
  all)
    run_abmp_ratios
    run_calib_sizes
    run_granularity
    ;;
  *)
    echo "Unknown RUN_SET=$RUN_SET. Use abmp_ratios, calib_sizes, granularity, daq_components, or all." >&2
    exit 2
    ;;
esac
