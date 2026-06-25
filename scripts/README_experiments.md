# Quant-dLLM Reproduction Experiment Scripts

These scripts are intended to be run after the current FP and Quant-DLLM full benchmarks finish.

## 1. Quant-DLLM ablation quantization

Quantize LLaDA-8B-Base variants for MMLU ablation:

```bash
cd /root/data-fs/Quant-dllm
conda activate qdlm

RUN_SET=abmp_ratios \
HF_HOME=/root/data-tmp/hf_eval \
bash scripts/exp_quantdllm_ablation_quantize.sh
```

Supported `RUN_SET` values:

- `abmp_ratios`: all-2bit / ABMP 0% / 5% / 10% / 15%.
- `calib_sizes`: C4 calibration sizes 64 / 128 / 256.
- `granularity`: weight / column / tile quantization granularity.
- `all`: runs all of the above.

Useful overrides:

```bash
DRY_RUN=1 bash scripts/exp_quantdllm_ablation_quantize.sh
START_BLOCK=0 END_BLOCK=1 bash scripts/exp_quantdllm_ablation_quantize.sh
MAX_MCS_SAMPLES=4 bash scripts/exp_quantdllm_ablation_quantize.sh
```

## 2. MMLU ablation evaluation

Evaluate all quantized ablation models under `outputs/ablations` on MMLU:

```bash
cd /root/data-fs/Quant-dllm
conda activate qdlm

HF_HOME=/root/data-tmp/hf_eval \
MC_NUM=128 \
MMLU_MC_NUM=128 \
bash scripts/exp_quantdllm_ablation_eval_mmlu.sh
```

Quick debug:

```bash
LIMIT=20 MC_NUM=16 MMLU_MC_NUM=16 bash scripts/exp_quantdllm_ablation_eval_mmlu.sh
```

## 3. GPTQ baseline

This uses QDLM's AutoGPTQ backend, but with a wrapper that fixes the LLaDA layer names and allows configurable output paths.

Quantize and evaluate GPTQ on paper multiple-choice tasks:

```bash
cd /root/data-fs/Quant-dllm
conda activate qdlm

CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/root/data-tmp/hf_eval \
WBITS=2 \
TASK_GROUP=paper_mc \
bash scripts/exp_gptq_llada8b_baseline.sh
```

Quantize only:

```bash
CUDA_VISIBLE_DEVICES=0 QUANTIZE_ONLY=1 bash scripts/exp_gptq_llada8b_baseline.sh
```

Evaluate an existing GPTQ model only:

```bash
CUDA_VISIBLE_DEVICES=0 EVAL_ONLY=1 TASKS=mmlu bash scripts/exp_gptq_llada8b_baseline.sh
```

Notes:

- AutoGPTQ is single-device by default here; use `CUDA_VISIBLE_DEVICES=0`.
- The wrapper uses C4 calibration with `CALIB_NSAMPLES=128` and `CALIB_SEQLEN=4096` by default.
- If GPTQ OOMs, first reduce `CALIB_SEQLEN` or use 3-bit/4-bit for a smoke test.

## 4. Cross-model template

Template for Dream/LLaDA variants:

```bash
cd /root/data-fs/Quant-dllm
conda activate qdlm

MODEL_NAME=dream7b_base \
MODEL_PATH=/path/to/Dream-7B-Base \
bash scripts/exp_cross_model_template.sh
```

Only use this after LLaDA-8B-Base results and ablations are stable.

## 5. Layer-wise / stage-wise Quant-dLLM pipelines

The reproduction now includes two calibration-aware PTQ pipelines under `tools/`:

- `tools/quant_layerwise_pipeline.py`: paper-aligned layer-wise pipeline. It computes MCS/SMCS statistics, ABMP bit allocation, and DAQ quantization for each target transformer block while propagating quantized hidden states block by block.
- `tools/quant_stagewise_pipeline.py`: diagnostic block-internal stage-wise variant. It quantizes `q/k/v`, `attn_out`, `ff_in`, and `ff_out` sequentially inside each transformer block and is useful for ablations, but the paper text is closer to the layer-wise pipeline.

Recommended paper-aligned pilot / full run uses tile-level DAQ and tile-level ABMP:

```bash
cd /root/data-fs/Quant-dllm

RUN_ID=layerwise_tiledaq_tileabmp_n128_s4096_t20_no_full_visible_end2_$(date +%Y%m%d_%H%M%S) \
PYTHON_BIN=/root/data-tmp/envs/qdlm/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2 \
END_BLOCK=2 \
RUN_EVAL=0 \
bash scripts/remote_layerwise_tile_pilot.sh
```

Run the full 32-block version with smoke evaluation:

```bash
cd /root/data-fs/Quant-dllm

RUN_ID=layerwise_tiledaq_tileabmp_n128_s4096_t20_no_full_visible_full_$(date +%Y%m%d_%H%M%S) \
PYTHON_BIN=/root/miniconda3/envs/qdlm/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2 \
END_BLOCK=32 \
RUN_EVAL=1 \
EVAL_TASKS=winogrande,piqa,arc_challenge,gsm8k \
EVAL_LIMIT=200 \
GPU_MEMORY=13GiB \
bash scripts/remote_layerwise_tile_pilot.sh
```

Stage-wise diagnostic run:

```bash
cd /root/data-fs/Quant-dllm

RUN_ID=stagewise_tiledaq_tileabmp_n128_s4096_t20_no_full_visible_$(date +%Y%m%d_%H%M%S) \
PYTHON_BIN=/root/miniconda3/envs/qdlm/bin/python \
CUDA_VISIBLE_DEVICES=0,1,2 \
DAQ_GRANULARITY=tile \
ABMP_GRANULARITY=tile \
ABMP_RANK_SCOPE=weight \
RUN_EVAL=1 \
EVAL_TASKS=winogrande,piqa,arc_challenge,gsm8k \
EVAL_LIMIT=200 \
GPU_MEMORY=13GiB \
bash scripts/local_stagewise_quant_eval.sh
```

Monitoring pattern:

```bash
cd /root/data-fs/Quant-dllm
LOG=$(ls -t logs/<run_id_prefix>*.log | head -1)
tail -f "$LOG"

PIDFILE=$(ls -t logs/<run_id_prefix>*.pid | head -1)
PID=$(cat "$PIDFILE")
ps -p "$PID" -o pid,ppid,sid,stat,pcpu,pmem,etime,cmd
nvidia-smi
```

Notes:

- `mcs_no_full_visible`, `nsamples=128`, `seqlen=4096`, and `num_steps=20` are the default paper-aligned calibration settings used by these scripts.
- On local 16GB V100s, `GPU_MEMORY=13GiB` is more stable during `device_map=auto` model loading than `15GiB`.
- `EVAL_LIMIT` is for smoke tests only and should not be compared directly with full benchmark results.

## 6. Current GPTQ-compensated DAQ mainline

The strongest paper-faithful reproduction line currently uses stage-wise statistics, column-group ABMP/DAQ, DOR, and GPTQ/OBC-style compensation. The detailed formulas and implementation mapping are in `docs/stagewise_gptq_daq_quantization.md`.

Launch the full WinoGrande reproduction run:

```bash
cd /root/data-fs/Quant-dllm

RUN_ID=stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHON_BIN=/root/miniconda3/envs/qdlm/bin/python \
C4_SOURCE=/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz \
MAX_TOKENS_PER_STATE=0 \
DAMPING=0.0 \
DAMPING_MODE=diag_mean \
DAMP_PERCENT=0.01 \
DAQ_GRANULARITY=column \
ABMP_GRANULARITY=column \
ABMP_RANK_SCOPE=weight \
GPTQ_COMPENSATION=1 \
DAQ_FULLS_REFINE=0 \
RUN_EVAL=1 \
EVAL_TASKS=winogrande \
EVAL_LIMIT=full \
bash scripts/local_stagewise_quant_eval.sh
```

Use `EVAL_LIMIT=200` only for smoke testing.
