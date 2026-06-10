#!/usr/bin/env bash
set -euo pipefail

cd /root/data-fs/Quant-dllm

PYTHON_BIN="${PYTHON_BIN:-/root/data-tmp/envs/qdlm/bin/python}"
MODEL_PATH="${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}"
C4_PATH="${C4_PATH:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"

NSAMPLES="${NSAMPLES:-128}"
SEQLEN="${SEQLEN:-4096}"
NUM_STEPS="${NUM_STEPS:-20}"
MAX_TOKENS_PER_STATE="${MAX_TOKENS_PER_STATE:-128}"
GPU_MEMORY="${GPU_MEMORY:-14GiB}"
CPU_MEMORY="${CPU_MEMORY:-120GiB}"
EVAL_GPU_MEMORY="${EVAL_GPU_MEMORY:-14GB}"

RUN_ID="${RUN_ID:-fullsmcs_col_col_full_n${NSAMPLES}_s${SEQLEN}_t${MAX_TOKENS_PER_STATE}_$(date +%Y%m%d_%H%M%S)}"
Z_DIR="${Z_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_column_full/${RUN_ID}_z_cache}"
OUT_DIR="${OUT_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_column_full/${RUN_ID}_llada8b_col_col}"
LOG_DIR="${LOG_DIR:-/root/data-fs/Quant-dllm/logs}"
BENCH_DIR="${BENCH_DIR:-/root/data-fs/Quant-dllm/benchmark_results}"

mkdir -p "$LOG_DIR" /root/data-fs/Quant-dllm/outputs/full_smcs_column_full "$BENCH_DIR"

printf '===== full-SMCS column/column run =====\n'
printf 'RUN_ID=%s\n' "$RUN_ID"
printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
printf 'MODEL_PATH=%s\n' "$MODEL_PATH"
printf 'C4_PATH=%s\n' "$C4_PATH"
printf 'Z_DIR=%s\n' "$Z_DIR"
printf 'OUT_DIR=%s\n' "$OUT_DIR"
printf 'NSAMPLES=%s SEQLEN=%s NUM_STEPS=%s MAX_TOKENS_PER_STATE=%s\n' "$NSAMPLES" "$SEQLEN" "$NUM_STEPS" "$MAX_TOKENS_PER_STATE"
printf 'CUDA_VISIBLE_DEVICES=%s GPU_MEMORY=%s CPU_MEMORY=%s\n' "$CUDA_VISIBLE_DEVICES" "$GPU_MEMORY" "$CPU_MEMORY"
printf '=======================================\n'

printf '\n===== Build sampled full-SMCS Z cache =====\n'
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u tools/build_full_smcs_z_cache.py \
  --model_path "$MODEL_PATH" \
  --source "$C4_PATH" \
  --z_cache_dir "$Z_DIR" \
  --start_block 0 \
  --nsamples "$NSAMPLES" \
  --seqlen "$SEQLEN" \
  --num-steps "$NUM_STEPS" \
  --max-tokens-per-state "$MAX_TOKENS_PER_STATE" \
  --mcs_no_full_visible \
  --damping 0.01 \
  --inverse-device cuda:0 \
  --gpu_memory "$GPU_MEMORY" \
  --cpu_memory "$CPU_MEMORY" \
  --offload_folder "/root/data-tmp/offload/${RUN_ID}_z" \
  --reuse_existing

printf '\n===== Quantize: column DAQ + column ABMP =====\n'
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -u /root/data-fs/Quant-dllm/main.py \
  --model_path "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --source "$C4_PATH" \
  --nsamples "$NSAMPLES" \
  --seed 42 \
  --seqlen "$SEQLEN" \
  --num-steps "$NUM_STEPS" \
  --gamma 0.25 \
  --group_size 128 \
  --schedule linear \
  --allocate_ratio 0.05 \
  --damping 0.01 \
  --daq_steps 10 \
  --imp_lamda 2.0 \
  --dor_mask_scope unit \
  --daq_component rsr_w_dor \
  --default_bits 2 \
  --reuse_z \
  --z_cache_dir "$Z_DIR" \
  --z_save_dtype fp32 \
  --save_dtype fp16 \
  --gpu_memory "$GPU_MEMORY" \
  --cpu_memory "$CPU_MEMORY" \
  --offload_folder "/root/data-tmp/offload/${RUN_ID}_quant" \
  --daq_granularity column \
  --abmp_granularity column \
  --abmp_rank_scope weight \
  --quant_block_size 128 \
  --daq_device cpu \
  --log_unit_interval 4 \
  --mcs_no_full_visible

printf '\n===== Evaluate WinoGrande limit=200 =====\n'
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
PYTHON_BIN="$PYTHON_BIN" \
RUN_ID="eval_${RUN_ID}_wino_limit200" \
MODEL_PATH="$OUT_DIR" \
OUTPUT_DIR="${BENCH_DIR}/eval_${RUN_ID}_wino_limit200" \
HF_HOME=/root/data-tmp/hf_eval \
HF_DATASETS_CACHE=/root/data-tmp/hf_eval/datasets \
HF_HUB_CACHE=/root/data-tmp/hf_eval/hub \
HF_MODULES_CACHE=/root/data-tmp/hf_eval/modules \
OFFLINE=1 \
TASKS=winogrande \
LIMIT=200 \
MC_NUM=128 \
GEN_BATCH_SIZE=1 \
LL_BATCH_SIZE=1 \
DTYPE=float16 \
USE_OFFLOAD=1 \
MAX_MEMORY_PER_GPU="$EVAL_GPU_MEMORY" \
MAX_CPU_MEMORY=0GB \
OFFLOAD_FOLDER="/root/data-tmp/offload/eval_${RUN_ID}_wino_limit200" \
bash /root/data-fs/Quant-dllm/llada_benchmark.sh

printf '\n===== DONE %s =====\n' "$RUN_ID"
printf 'Z_DIR=%s\n' "$Z_DIR"
printf 'OUT_DIR=%s\n' "$OUT_DIR"
printf 'Wino result dir=%s\n' "${BENCH_DIR}/eval_${RUN_ID}_wino_limit200"
