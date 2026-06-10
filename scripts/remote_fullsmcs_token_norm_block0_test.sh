#!/usr/bin/env bash
set -euo pipefail

cd /root/data-fs/Quant-dllm

PYTHON_BIN=${PYTHON_BIN:-/root/data-tmp/envs/qdlm/bin/python}
MODEL_PATH=${MODEL_PATH:-/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base}
C4_SOURCE=${C4_SOURCE:-/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz}

RUN_ID=${RUN_ID:-fullsmcs_token_norm_block0_t0_$(date +%Y%m%d_%H%M%S)}
Z_CACHE_DIR=${Z_CACHE_DIR:-/root/data-fs/Quant-dllm/outputs/full_smcs_token_norm_check/${RUN_ID}_z_cache}
LOG_DIR=/root/data-fs/Quant-dllm/logs
LOG_FILE=${LOG_FILE:-${LOG_DIR}/${RUN_ID}.log}
OFFLOAD_FOLDER=${OFFLOAD_FOLDER:-/root/data-tmp/offload/${RUN_ID}}

mkdir -p "$LOG_DIR" "$OFFLOAD_FOLDER" "$(dirname "$Z_CACHE_DIR")"

{
  echo "===== Stop old full-SMCS jobs ====="
  ps -eo pid,ppid,sid,etime,%cpu,%mem,cmd \
    | grep -E 'fullsmcs_col_col_full|full_smcs_column_full|build_full_smcs_z_cache.py' \
    | grep -v grep || true

  pkill -TERM -f 'fullsmcs_col_col_full' || true
  pkill -TERM -f 'full_smcs_column_full' || true
  pkill -TERM -f 'build_full_smcs_z_cache.py.*full_smcs_column_full' || true

  sleep 10

  if ps -eo pid,cmd \
      | grep -E 'fullsmcs_col_col_full|full_smcs_column_full|build_full_smcs_z_cache.py.*full_smcs_column_full' \
      | grep -v grep; then
    echo "Old process still alive after TERM; sending KILL."
    pkill -KILL -f 'fullsmcs_col_col_full' || true
    pkill -KILL -f 'full_smcs_column_full' || true
    pkill -KILL -f 'build_full_smcs_z_cache.py.*full_smcs_column_full' || true
  fi

  echo
  echo "===== New block0 full-SMCS token-normalized Z test ====="
  echo "RUN_ID=$RUN_ID"
  echo "PYTHON_BIN=$PYTHON_BIN"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "C4_SOURCE=$C4_SOURCE"
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "LOG_FILE=$LOG_FILE"
  echo "OFFLOAD_FOLDER=$OFFLOAD_FOLDER"
  echo

  CUDA_VISIBLE_DEVICES=0,1,2 \
  "$PYTHON_BIN" -u /root/data-fs/Quant-dllm/tools/build_full_smcs_z_cache.py \
    --model_path "$MODEL_PATH" \
    --source "$C4_SOURCE" \
    --z_cache_dir "$Z_CACHE_DIR" \
    --start_block 0 \
    --end_block 1 \
    --nsamples 128 \
    --seed 42 \
    --seqlen 4096 \
    --num-steps 20 \
    --gamma 0.25 \
    --schedule linear \
    --max-tokens-per-state 0 \
    --mcs_no_full_visible \
    --damping 0.01 \
    --mcs-normalization tokens \
    --inverse-diag-floor 1e-12 \
    --inverse-device cuda:0 \
    --gpu_memory 15GiB \
    --cpu_memory 120GiB \
    --offload_folder "$OFFLOAD_FOLDER"

  echo
  echo "===== Finite check for generated block_00 ====="

  "$PYTHON_BIN" - <<PY
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

z_cache_dir = Path("$Z_CACHE_DIR")
path = z_cache_dir / "block_00.safetensors"
if not path.exists():
    raise SystemExit(f"Missing expected Z file: {path}")

sd = load_file(str(path), device="cpu")
summary = {}
bad = {}

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
        })
    summary[name] = item
    if nonfinite:
        bad[name] = item

print(json.dumps({
    "z_cache_dir": str(z_cache_dir),
    "file": str(path),
    "bad_tensor_count": len(bad),
    "bad": bad,
    "summary": summary,
}, indent=2))

if bad:
    raise SystemExit("FAILED: non-finite values found in generated Z cache.")

print("PASS: all tensors in block_00 are finite.")
PY

  echo
  echo "Done."
  echo "Z_CACHE_DIR=$Z_CACHE_DIR"
  echo "LOG_FILE=$LOG_FILE"
} 2>&1 | tee "$LOG_FILE"
