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
