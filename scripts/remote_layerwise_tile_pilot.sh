#!/usr/bin/env bash
set -euo pipefail

cd /root/data-fs/Quant-dllm

PYTHON_BIN=${PYTHON_BIN:-/root/data-tmp/envs/qdlm/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}
C4_SOURCE=${C4_SOURCE:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export CUDA_VISIBLE_DEVICES

END_BLOCK=${END_BLOCK:-2}
RUN_ID=${RUN_ID:-layerwise_tiledaq_tileabmp_n128_s4096_t20_no_full_visible_end${END_BLOCK}_$(date +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-/root/data-fs/Quant-dllm/outputs/${RUN_ID}}
Z_CACHE_DIR=${Z_CACHE_DIR:-${OUTPUT_DIR}/z_cache}
HIDDEN_CACHE_DIR=${HIDDEN_CACHE_DIR:-/root/data-tmp/layerwise_hidden_cache/${RUN_ID}}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-/root/data-tmp/offload/${RUN_ID}}
LOG_DIR=${LOG_DIR:-/root/data-fs/Quant-dllm/logs}
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}
PIDFILE=${PIDFILE:-${LOG_DIR}/${RUN_ID}.pid}

NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-4096}
NUM_STEPS=${NUM_STEPS:-20}
GAMMA=${GAMMA:-0.25}
HIDDEN_CHUNK_SIZE=${HIDDEN_CHUNK_SIZE:-1}
MAX_MCS_SAMPLES=${MAX_MCS_SAMPLES:-}
MAX_TOKENS_PER_STATE=${MAX_TOKENS_PER_STATE:-0}
DAMPING=${DAMPING:-0.01}
INVERSE_DIAG_FLOOR=${INVERSE_DIAG_FLOOR:-1e-12}
INVERSE_DEVICE=${INVERSE_DEVICE:-cuda:0}
GPU_MEMORY=${GPU_MEMORY:-15GiB}
CPU_MEMORY=${CPU_MEMORY:-120GiB}

DAQ_GRANULARITY=${DAQ_GRANULARITY:-tile}
ABMP_GRANULARITY=${ABMP_GRANULARITY:-tile}
ABMP_RANK_SCOPE=${ABMP_RANK_SCOPE:-weight}
QUANT_BLOCK_SIZE=${QUANT_BLOCK_SIZE:-128}
ALLOCATE_RATIO=${ALLOCATE_RATIO:-0.05}
DEFAULT_BITS=${DEFAULT_BITS:-2}
DAQ_STEPS=${DAQ_STEPS:-10}
IMP_LAMDA=${IMP_LAMDA:-2.0}
DAQ_DEVICE=${DAQ_DEVICE:-cpu}
DOR_MASK_SCOPE=${DOR_MASK_SCOPE:-unit}
DAQ_COMPONENT=${DAQ_COMPONENT:-rsr_w_dor}

RUN_EVAL=${RUN_EVAL:-0}
EVAL_TASKS=${EVAL_TASKS:-winogrande,piqa,arc_challenge,gsm8k,bbh_zeroshot}
EVAL_LIMIT=${EVAL_LIMIT:-200}
HF_HOME=${HF_HOME:-/root/data-tmp/hf_eval}

mkdir -p "$LOG_DIR" "$OFFLOAD_FOLDER" "$HIDDEN_CACHE_DIR" "$OUTPUT_DIR"
echo $$ > "$PIDFILE"

{
  echo "===== Layer-wise tile Quant-dLLM pilot ====="
  echo "RUN_ID=$RUN_ID"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "C4_SOURCE=$C4_SOURCE"
  echo "OUTPUT_DIR=$OUTPUT_DIR"
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "HIDDEN_CACHE_DIR=$HIDDEN_CACHE_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "END_BLOCK=$END_BLOCK"
  echo "NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN NUM_STEPS=$NUM_STEPS MAX_MCS_SAMPLES=${MAX_MCS_SAMPLES:-<none>}"
  echo "DAQ_GRANULARITY=$DAQ_GRANULARITY ABMP_GRANULARITY=$ABMP_GRANULARITY ABMP_RANK_SCOPE=$ABMP_RANK_SCOPE"
  echo "Started: $(date)"

  quant_cmd=(
    "$PYTHON_BIN" -u /root/data-fs/Quant-dllm/tools/quant_layerwise_pipeline.py
    --model_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --source "$C4_SOURCE"
    --nsamples "$NSAMPLES"
    --seed 42
    --seqlen "$SEQLEN"
    --num-steps "$NUM_STEPS"
    --gamma "$GAMMA"
    --group_size 128
    --schedule linear
    --mcs_no_full_visible
    --mcs_normalization tokens
    --max_tokens_per_state "$MAX_TOKENS_PER_STATE"
    --damping "$DAMPING"
    --inverse_diag_floor "$INVERSE_DIAG_FLOOR"
    --inverse_device "$INVERSE_DEVICE"
    --allocate_ratio "$ALLOCATE_RATIO"
    --daq_steps "$DAQ_STEPS"
    --imp_lamda "$IMP_LAMDA"
    --dor_mask_scope "$DOR_MASK_SCOPE"
    --daq_component "$DAQ_COMPONENT"
    --default_bits "$DEFAULT_BITS"
    --z_save_dtype fp32
    --save_dtype fp16
    --daq_granularity "$DAQ_GRANULARITY"
    --abmp_granularity "$ABMP_GRANULARITY"
    --abmp_rank_scope "$ABMP_RANK_SCOPE"
    --quant_block_size "$QUANT_BLOCK_SIZE"
    --daq_device "$DAQ_DEVICE"
    --gpu_memory "$GPU_MEMORY"
    --cpu_memory "$CPU_MEMORY"
    --offload_folder "$OFFLOAD_FOLDER"
    --z_cache_dir "$Z_CACHE_DIR"
    --hidden_cache_dir "$HIDDEN_CACHE_DIR"
    --hidden_chunk_size "$HIDDEN_CHUNK_SIZE"
    --end_block "$END_BLOCK"
  )
  if [ -n "${MAX_MCS_SAMPLES:-}" ]; then
    quant_cmd+=(--max_mcs_samples "$MAX_MCS_SAMPLES")
  fi

  echo "Command: ${quant_cmd[*]}"
  "${quant_cmd[@]}"

  echo "Quantization finished: $(date)"

  if [ "$RUN_EVAL" = "1" ]; then
    if [ "$END_BLOCK" != "32" ]; then
      echo "WARNING: RUN_EVAL=1 with END_BLOCK=$END_BLOCK evaluates a partially quantized model."
    fi
    if [ "$EVAL_LIMIT" = "full" ] || [ "$EVAL_LIMIT" = "none" ]; then
      EVAL_LIMIT_VALUE=""
      EVAL_LIMIT_TAG="full"
    else
      EVAL_LIMIT_VALUE="$EVAL_LIMIT"
      EVAL_LIMIT_TAG="limit${EVAL_LIMIT}"
    fi
    EVAL_OUTPUT_DIR=/root/data-fs/Quant-dllm/benchmark_results/${RUN_ID}_${EVAL_LIMIT_TAG}
    EVAL_OFFLOAD_FOLDER=/root/data-tmp/offload/eval_${RUN_ID}_${EVAL_LIMIT_TAG}
    mkdir -p "$EVAL_OUTPUT_DIR" "$EVAL_OFFLOAD_FOLDER"
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_ID="${RUN_ID}_${EVAL_LIMIT_TAG}" \
    MODEL_PATH="$OUTPUT_DIR" \
    OUTPUT_DIR="$EVAL_OUTPUT_DIR" \
    HF_HOME="$HF_HOME" \
    HF_DATASETS_CACHE="$HF_HOME/datasets" \
    HF_HUB_CACHE="$HF_HOME/hub" \
    HF_MODULES_CACHE="$HF_HOME/modules" \
    OFFLINE=1 \
    TASKS="$EVAL_TASKS" \
    LIMIT="$EVAL_LIMIT_VALUE" \
    MC_NUM=128 \
    GEN_BATCH_SIZE=1 \
    LL_BATCH_SIZE=1 \
    DTYPE=float16 \
    USE_OFFLOAD=1 \
    MAX_MEMORY_PER_GPU=15GB \
    MAX_CPU_MEMORY=0GB \
    OFFLOAD_FOLDER="$EVAL_OFFLOAD_FOLDER" \
    bash /root/data-fs/Quant-dllm/llada_benchmark.sh
  fi

  echo "Finished: $(date)"
} 2>&1 | tee "$LOG_FILE"
