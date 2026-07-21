# Quant-dLLM Reproduction

This repository contains the cleaned reproduction path used for Quant-dLLM experiments on LLaDA and Dream diffusion language models. It intentionally keeps one quantization pipeline and excludes the new research extensions maintained in the sibling `Quant-dllm-research` repository.

## Mainline

The canonical path is:

1. Build MCS calibration states from C4.
2. Collect stage-wise SMCS statistics from the current quantized trajectory.
3. Construct distilled importance matrices `Z` from the full SMCS inverse diagonal.
4. Allocate 1/2/3-bit column groups with ABMP.
5. Quantize each group with DAQ/DOR.
6. Apply GPTQ/OBC-style compensation to later input-channel groups.
7. Propagate the quantized transformer-block output to the next block.

The verified defaults are `nsamples=128`, `seqlen=4096`, `num_steps=20`, no fully visible MCS state, token-normalized SMCS, column groups of 128, per-weight ABMP, `allocate_ratio=0.05`, DAQ steps 10, and GPTQ compensation.

## Layout

- `ABMP/`: precision allocation.
- `DAQ/`: binary residual quantization and DOR optimization.
- `MCS/`: masked calibration state construction.
- `utils/smcs.py`: SMCS hooks and distilled `Z` construction.
- `tools/pipeline_common.py`: model-family and hidden-cache utilities.
- `tools/quant_stagewise_pipeline.py`: the single production quantization pipeline.
- `scripts/run_quantization.sh`: reproducible launcher with optional post-quantization evaluation.
- `llada_benchmark.sh`: LLaDA evaluation through the QDLM harness.
- `dream_benchmark.sh`: Dream evaluation through the QDLM harness.
- `tools/probe_dllm_model.py`: local model compatibility probe.

## Quick Start

```bash
cd /root/data-fs/Quant-dllm-repro

RUN_ID=llada8b_repro_$(date +%Y%m%d_%H%M%S) \
MODEL_PATH=/path/to/LLaDA-8B-Base \
C4_SOURCE=/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz \
CUDA_VISIBLE_DEVICES=0,1,2 \
RUN_EVAL=1 \
EVAL_TASKS=winogrande,piqa,arc_easy,arc_challenge,bbh_zeroshot \
EVAL_LIMIT=200 \
bash scripts/run_quantization.sh
```

Use `EVAL_LIMIT=full` for complete benchmarks. Limited runs are smoke tests and must not be compared directly with full benchmark values.

For Dream, set `MODEL_PATH` to the local Dream snapshot and optionally set `MODEL_FAMILY=dream`. The launcher otherwise detects the family from the saved config.

## Reproduction Boundaries

The canonical launcher does not enable Dream GQA-aware bit reallocation, concentrated/sparse block-internal splitting, FP denoising-trajectory calibration, output-Jacobian allocation, activation-diagonal DAQ, or fixed-B full-S scale refinement. Active experiments and launchers for those methods belong in `Quant-dllm-research`; a few low-level compatibility hooks remain in the shared core for checkpoint continuity.

Detailed formulas and implementation mapping are in `docs/stagewise_gptq_daq_quantization.md`.
