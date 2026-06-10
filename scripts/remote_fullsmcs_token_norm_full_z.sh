#!/usr/bin/env bash
set -euo pipefail

cd /root/data-fs/Quant-dllm

PYTHON_BIN=${PYTHON_BIN:-/root/data-tmp/envs/qdlm/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}
C4_SOURCE=${C4_SOURCE:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}

RUN_ID=${RUN_ID:-fullsmcs_token_norm_full_n128_s4096_t0}
Z_CACHE_DIR=${Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_token_norm_full/${RUN_ID}_z_cache}
SEED_Z_CACHE_DIR=${SEED_Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_token_norm_check/fullsmcs_token_norm_block0_t0_20260530_125917_z_cache}
LOG_DIR=/root/data-fs/Quant-dllm/logs
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}_$(date +%Y%m%d_%H%M%S).log}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-/root/data-tmp/offload/${RUN_ID}}

NSAMPLES=${NSAMPLES:-128}
SEQLEN=${SEQLEN:-4096}
NUM_STEPS=${NUM_STEPS:-20}
MAX_TOKENS_PER_STATE=${MAX_TOKENS_PER_STATE:-0}
DAMPING=${DAMPING:-0.01}
INVERSE_DIAG_FLOOR=${INVERSE_DIAG_FLOOR:-1e-12}
GPU_MEMORY=${GPU_MEMORY:-15GiB}
CPU_MEMORY=${CPU_MEMORY:-120GiB}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2}
RUN_FINITE_CHECK=${RUN_FINITE_CHECK:-1}
STOP_OLD_COLUMN_FULL=${STOP_OLD_COLUMN_FULL:-0}

mkdir -p "$LOG_DIR" "$OFFLOAD_FOLDER" "$Z_CACHE_DIR"

{
  echo "===== Full token-normalized full-SMCS Z generation ====="
  echo "RUN_ID=$RUN_ID"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "C4_SOURCE=$C4_SOURCE"
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "SEED_Z_CACHE_DIR=$SEED_Z_CACHE_DIR"
  echo "LOG_FILE=$LOG_FILE"
  echo "OFFLOAD_FOLDER=$OFFLOAD_FOLDER"
  echo "CUDA_DEVICES=$CUDA_DEVICES"
  echo "NSAMPLES=$NSAMPLES"
  echo "SEQLEN=$SEQLEN"
  echo "NUM_STEPS=$NUM_STEPS"
  echo "MAX_TOKENS_PER_STATE=$MAX_TOKENS_PER_STATE"
  echo "DAMPING=$DAMPING"
  echo "INVERSE_DIAG_FLOOR=$INVERSE_DIAG_FLOOR"
  echo

  if [ "$STOP_OLD_COLUMN_FULL" = "1" ]; then
    echo "===== Stop old column/full-SMCS jobs only ====="
    ps -eo pid,ppid,sid,etime,%cpu,%mem,cmd \
      | grep -E 'fullsmcs_col_col_full|full_smcs_column_full|build_full_smcs_z_cache.py.*full_smcs_column_full' \
      | grep -v grep || true

    pkill -TERM -f 'fullsmcs_col_col_full' || true
    pkill -TERM -f 'full_smcs_column_full' || true
    pkill -TERM -f 'build_full_smcs_z_cache.py.*full_smcs_column_full' || true
    sleep 5
  fi

  if [ -f "$SEED_Z_CACHE_DIR/block_00.safetensors" ] && [ ! -f "$Z_CACHE_DIR/block_00.safetensors" ]; then
    echo "Seeding validated block_00 from: $SEED_Z_CACHE_DIR"
    cp "$SEED_Z_CACHE_DIR/block_00.safetensors" "$Z_CACHE_DIR/block_00.safetensors"
  fi

  echo
  echo "Existing block files before run:"
  find "$Z_CACHE_DIR" -maxdepth 1 -name 'block_*.safetensors' -printf '%f\n' | sort || true
  echo

  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  "$PYTHON_BIN" -u /root/data-fs/Quant-dllm/tools/build_full_smcs_z_cache.py \
    --model_path "$MODEL_PATH" \
    --source "$C4_SOURCE" \
    --z_cache_dir "$Z_CACHE_DIR" \
    --start_block 0 \
    --end_block 32 \
    --nsamples "$NSAMPLES" \
    --seed 42 \
    --seqlen "$SEQLEN" \
    --num-steps "$NUM_STEPS" \
    --gamma 0.25 \
    --schedule linear \
    --max-tokens-per-state "$MAX_TOKENS_PER_STATE" \
    --mcs_no_full_visible \
    --damping "$DAMPING" \
    --mcs-normalization tokens \
    --inverse-diag-floor "$INVERSE_DIAG_FLOOR" \
    --inverse-device cuda:0 \
    --gpu_memory "$GPU_MEMORY" \
    --cpu_memory "$CPU_MEMORY" \
    --offload_folder "$OFFLOAD_FOLDER" \
    --reuse_existing

  echo
  echo "Generated block files after run:"
  find "$Z_CACHE_DIR" -maxdepth 1 -name 'block_*.safetensors' -printf '%f\n' | sort || true
  echo

  if [ "$RUN_FINITE_CHECK" = "1" ]; then
    echo "===== Finite check for all generated blocks ====="
    "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

z_cache_dir = Path("$Z_CACHE_DIR")
expected = [z_cache_dir / f"block_{idx:02d}.safetensors" for idx in range(32)]
missing = [str(path) for path in expected if not path.exists()]
if missing:
    raise SystemExit("Missing block files:\n" + "\n".join(missing))

bad = {}
summary = {}
for block_idx, path in enumerate(expected):
    sd = load_file(str(path), device="cpu")
    block_bad = {}
    block_summary = {}
    for name, tensor in sd.items():
        x = tensor.float()
        finite = torch.isfinite(x)
        nonfinite = int((~finite).sum().item())
        item = {
            "shape": list(x.shape),
            "numel": x.numel(),
            "nonfinite": nonfinite,
            "nan": int(torch.isnan(x).sum().item()),
            "posinf": int(torch.isposinf(x).sum().item()),
            "neginf": int(torch.isneginf(x).sum().item()),
        }
        if finite.any():
            xf = x[finite]
            item.update({
                "finite_min": float(xf.min().item()),
                "finite_max": float(xf.max().item()),
                "finite_mean": float(xf.mean().item()),
                "finite_sum": float(xf.sum().item()),
            })
        block_summary[name] = item
        if nonfinite:
            block_bad[name] = item
    summary[f"block_{block_idx:02d}"] = block_summary
    if block_bad:
        bad[f"block_{block_idx:02d}"] = block_bad

out = {
    "z_cache_dir": str(z_cache_dir),
    "blocks_checked": 32,
    "bad_block_count": len(bad),
    "bad": bad,
    "summary": summary,
}
report_path = z_cache_dir / "finite_check_summary.json"
report_path.write_text(json.dumps(out, indent=2))
print(json.dumps({
    "z_cache_dir": str(z_cache_dir),
    "blocks_checked": 32,
    "bad_block_count": len(bad),
    "report_path": str(report_path),
}, indent=2))

if bad:
    raise SystemExit("FAILED: non-finite values found in generated Z cache.")
print("PASS: all tensors in all 32 blocks are finite.")
PY
  fi

  echo
  echo "Done."
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "LOG_FILE=$LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
