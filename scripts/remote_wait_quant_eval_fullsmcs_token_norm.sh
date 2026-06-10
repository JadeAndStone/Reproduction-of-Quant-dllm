#!/usr/bin/env bash
set -euo pipefail

cd /root/data-fs/Quant-dllm

PYTHON_BIN=${PYTHON_BIN:-/root/data-tmp/envs/qdlm/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}
C4_SOURCE=${C4_SOURCE:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}

Z_RUN_ID=${Z_RUN_ID:-fullsmcs_token_norm_full_n128_s4096_t0}
Z_CACHE_DIR=${Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_token_norm_full/${Z_RUN_ID}_z_cache}

RUN_ID=${RUN_ID:-fullsmcs_token_norm_tiledaq_colabmp_n128_s4096_t0}
QUANT_OUTPUT_DIR=${QUANT_OUTPUT_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_token_norm_quant/${RUN_ID}}
LOG_DIR=/root/data-fs/Quant-dllm/logs
QUANT_LOG=${QUANT_LOG:-${LOG_DIR}/quant_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log}
EVAL_LOG=${EVAL_LOG:-${LOG_DIR}/eval_${RUN_ID}_$(date +%Y%m%d_%H%M%S).log}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-/root/data-tmp/offload/quant_${RUN_ID}}
EVAL_OFFLOAD_FOLDER=${EVAL_OFFLOAD_FOLDER:-/root/data-tmp/offload/eval_${RUN_ID}}

WAIT_INTERVAL=${WAIT_INTERVAL:-300}
MAX_WAIT_HOURS=${MAX_WAIT_HOURS:-96}
REQUIRE_FINITE_REPORT=${REQUIRE_FINITE_REPORT:-1}
FORCE_QUANT=${FORCE_QUANT:-0}

NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-4096}
NUM_STEPS=${NUM_STEPS:-20}
DAMPING=${DAMPING:-0.01}
INVERSE_DIAG_FLOOR=${INVERSE_DIAG_FLOOR:-1e-12}
GPU_MEMORY=${GPU_MEMORY:-15GiB}
CPU_MEMORY=${CPU_MEMORY:-120GiB}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2}

EVAL_TASKS=${EVAL_TASKS:-winogrande,piqa,arc_challenge,gsm8k}
EVAL_LIMIT=${EVAL_LIMIT:-200}
RUN_BBH=${RUN_BBH:-0}
BBH_LIMIT=${BBH_LIMIT:-20}
MC_NUM=${MC_NUM:-128}
DTYPE=${DTYPE:-float16}
MAX_MEMORY_PER_GPU=${MAX_MEMORY_PER_GPU:-15GB}
MAX_CPU_MEMORY=${MAX_CPU_MEMORY:-0GB}
HF_HOME=${HF_HOME:-/root/data-tmp/hf_eval}
OFFLINE=${OFFLINE:-1}

mkdir -p "$LOG_DIR" "$(dirname "$QUANT_OUTPUT_DIR")" "$OFFLOAD_FOLDER" "$EVAL_OFFLOAD_FOLDER"

wait_for_z_cache() {
  local deadline=$(( $(date +%s) + MAX_WAIT_HOURS * 3600 ))
  while true; do
    local missing=0
    for idx in $(seq -w 0 31); do
      if [ ! -f "$Z_CACHE_DIR/block_${idx}.safetensors" ]; then
        missing=1
        break
      fi
    done

    if [ "$missing" = "0" ]; then
      if [ "$REQUIRE_FINITE_REPORT" = "1" ]; then
        if [ -f "$Z_CACHE_DIR/finite_check_summary.json" ]; then
          "$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$Z_CACHE_DIR") / "finite_check_summary.json"
data = json.loads(p.read_text())
bad = int(data.get("bad_block_count", -1))
if bad != 0:
    raise SystemExit(f"finite_check_summary reports bad_block_count={bad}")
print("finite_check_summary OK")
PY
          return 0
        fi
        echo "All block files exist, waiting for finite_check_summary.json ..."
      else
        return 0
      fi
    else
      echo "Waiting for complete Z cache at $Z_CACHE_DIR ..."
    fi

    if [ "$(date +%s)" -gt "$deadline" ]; then
      echo "Timed out waiting for Z cache after ${MAX_WAIT_HOURS} hours." >&2
      exit 1
    fi
    sleep "$WAIT_INTERVAL"
  done
}

{
  echo "===== Wait for full Z cache ====="
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "REQUIRE_FINITE_REPORT=$REQUIRE_FINITE_REPORT"
  wait_for_z_cache

  echo
  echo "===== Quantization config ====="
  echo "RUN_ID=$RUN_ID"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "C4_SOURCE=$C4_SOURCE"
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "QUANT_OUTPUT_DIR=$QUANT_OUTPUT_DIR"
  echo "QUANT_LOG=$QUANT_LOG"
  echo "OFFLOAD_FOLDER=$OFFLOAD_FOLDER"
  echo "CUDA_DEVICES=$CUDA_DEVICES"
  echo

  if [ "$FORCE_QUANT" != "1" ] && [ -f "$QUANT_OUTPUT_DIR/model.safetensors.index.json" ]; then
    echo "Quantized model already exists, skip quantization: $QUANT_OUTPUT_DIR"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    "$PYTHON_BIN" -u /root/data-fs/Quant-dllm/main.py \
      --model_path "$MODEL_PATH" \
      --output_dir "$QUANT_OUTPUT_DIR" \
      --source "$C4_SOURCE" \
      --nsamples "$NSAMPLES" \
      --seed 42 \
      --seqlen "$SEQLEN" \
      --num-steps "$NUM_STEPS" \
      --gamma 0.25 \
      --group_size 128 \
      --schedule linear \
      --allocate_ratio 0.05 \
      --damping "$DAMPING" \
      --mcs_normalization tokens \
      --inverse_diag_floor "$INVERSE_DIAG_FLOOR" \
      --daq_steps 10 \
      --imp_lamda 2.0 \
      --dor_mask_scope unit \
      --daq_component rsr_w_dor \
      --default_bits 2 \
      --reuse_z \
      --z_cache_dir "$Z_CACHE_DIR" \
      --z_save_dtype fp32 \
      --save_dtype fp16 \
      --gpu_memory "$GPU_MEMORY" \
      --cpu_memory "$CPU_MEMORY" \
      --offload_folder "$OFFLOAD_FOLDER" \
      --daq_granularity tile \
      --abmp_granularity column \
      --abmp_rank_scope weight \
      --quant_block_size 128 \
      --daq_device cpu \
      --log_unit_interval 256 \
      --mcs_no_full_visible
  fi
} 2>&1 | tee "$QUANT_LOG"

{
  echo "===== Eval config ====="
  echo "RUN_ID=$RUN_ID"
  echo "QUANT_OUTPUT_DIR=$QUANT_OUTPUT_DIR"
  echo "EVAL_TASKS=$EVAL_TASKS"
  echo "EVAL_LIMIT=$EVAL_LIMIT"
  echo "RUN_BBH=$RUN_BBH"
  echo "BBH_LIMIT=$BBH_LIMIT"
  echo "EVAL_LOG=$EVAL_LOG"
  echo

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  PYTHON_BIN="$PYTHON_BIN" \
  RUN_ID="eval_${RUN_ID}_limit${EVAL_LIMIT}" \
  MODEL_PATH="$QUANT_OUTPUT_DIR" \
  OUTPUT_DIR="/root/data-fs/Quant-dllm/benchmark_results/eval_${RUN_ID}_limit${EVAL_LIMIT}" \
  HF_HOME="$HF_HOME" \
  HF_DATASETS_CACHE="$HF_HOME/datasets" \
  HF_HUB_CACHE="$HF_HOME/hub" \
  HF_MODULES_CACHE="$HF_HOME/modules" \
  OFFLINE="$OFFLINE" \
  TASKS="$EVAL_TASKS" \
  LIMIT="$EVAL_LIMIT" \
  MC_NUM="$MC_NUM" \
  GEN_BATCH_SIZE=1 \
  LL_BATCH_SIZE=1 \
  DTYPE="$DTYPE" \
  USE_OFFLOAD=1 \
  MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" \
  MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
  OFFLOAD_FOLDER="$EVAL_OFFLOAD_FOLDER" \
  bash /root/data-fs/Quant-dllm/llada_benchmark.sh

  if [ "$RUN_BBH" = "1" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_ID="eval_${RUN_ID}_bbh_limit${BBH_LIMIT}" \
    MODEL_PATH="$QUANT_OUTPUT_DIR" \
    OUTPUT_DIR="/root/data-fs/Quant-dllm/benchmark_results/eval_${RUN_ID}_bbh_limit${BBH_LIMIT}" \
    HF_HOME="$HF_HOME" \
    HF_DATASETS_CACHE="$HF_HOME/datasets" \
    HF_HUB_CACHE="$HF_HOME/hub" \
    HF_MODULES_CACHE="$HF_HOME/modules" \
    OFFLINE="$OFFLINE" \
    TASKS="bbh_zeroshot" \
    LIMIT="$BBH_LIMIT" \
    MC_NUM="$MC_NUM" \
    GEN_BATCH_SIZE=1 \
    LL_BATCH_SIZE=1 \
    DTYPE="$DTYPE" \
    USE_OFFLOAD=1 \
    MAX_MEMORY_PER_GPU="$MAX_MEMORY_PER_GPU" \
    MAX_CPU_MEMORY="$MAX_CPU_MEMORY" \
    OFFLOAD_FOLDER="${EVAL_OFFLOAD_FOLDER}_bbh" \
    bash /root/data-fs/Quant-dllm/llada_benchmark.sh
  fi

  echo
  echo "Done."
  echo "QUANT_OUTPUT_DIR=$QUANT_OUTPUT_DIR"
  echo "QUANT_LOG=$QUANT_LOG"
  echo "EVAL_LOG=$EVAL_LOG"
} 2>&1 | tee "$EVAL_LOG"
