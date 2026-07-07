# Reproduction of Quant-dLLM

This repository contains a reproduction and diagnostic implementation of
**Quant-dLLM**, a post-training quantization method for diffusion large language
models. The current target model is **LLaDA-8B-Base** under **2-bit weight-only
PTQ**.

The strongest reproduction line in this repo uses:

- masked calibration simulation (MCS) for diffusion-style masked inputs;
- SMCS / distilled importance matrix `Z`;
- adaptive blockwise mixed precision (ABMP);
- DAQ/DOR binary decomposition quantization;
- stage-wise statistics inside each transformer block;
- GPTQ/OBC-style column-group error compensation.

The detailed project presentation is available at
[`docs/reproduction_project_structure_beamer.pdf`](docs/reproduction_project_structure_beamer.pdf).
The mainline implementation notes are in
[`docs/stagewise_gptq_daq_quantization.md`](docs/stagewise_gptq_daq_quantization.md).

## Status

The latest verified mainline run is:

```text
stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_20260615_023437
```

This run uses column-group DAQ/ABMP plus GPTQ-style compensation. It improved
the early WinoGrande reproduction from about `0.6267` to `0.6614`.

## Method Overview

The reproduction pipeline follows the paper's core route:

```text
MCS calibration
  -> hidden-state cache
  -> stage-wise SMCS statistics
  -> importance matrix Z
  -> ABMP bit allocation
  -> DAQ/DOR quantization
  -> GPTQ compensation
  -> checkpoint save and benchmark evaluation
```

Inside each transformer block, the stage-wise pipeline quantizes modules in the
following order:

1. `qkv`: `q_proj`, `k_proj`, `v_proj`
2. `attn_out`: `attn_out`
3. `ff_in`: `ff_proj`, `up_proj`
4. `ff_out`: `ff_out`

Each stage collects statistics from the current partially quantized model state
and immediately writes quantized weights back to the in-memory model. Later
stages therefore see the effect of earlier quantization.

## Main Configuration

The verified mainline configuration is:

| Item | Value |
| --- | --- |
| Calibration samples | `nsamples=128` |
| Sequence length | `seqlen=4096` |
| MCS steps | `num_steps=20` |
| Mask schedule | `linear` |
| Fully visible endpoint | disabled by `mcs_no_full_visible` |
| MCS normalization | `tokens` |
| Token subsampling | `max_tokens_per_state=0` |
| SMCS damping | `damping_mode=diag_mean`, `damp_percent=0.01`, `damping=0.0` |
| DAQ granularity | `column` |
| ABMP granularity | `column` |
| ABMP rank scope | `weight` |
| Group size | `quant_block_size=128` |
| Base precision | `default_bits=2` |
| Mixed precision ratio | `allocate_ratio=0.05` |
| DAQ component | `rsr_w_dor` |
| DAQ steps | `daq_steps=10` |
| DOR weight | `imp_lamda=2.0` |
| DOR mask scope | `unit` |
| GPTQ compensation | enabled |
| Row centering | disabled |

## Repository Structure

```text
.
|-- MCS/                         # masked calibration simulation
|-- DAQ/                         # DAQ/DOR binary decomposition quantizer
|-- ABMP/                        # early ABMP interfaces
|-- utils/                       # SMCS / Z construction helpers
|-- tools/
|   |-- quant_stagewise_pipeline.py
|   |-- quant_layerwise_pipeline.py
|   |-- build_full_smcs_z_cache.py
|   |-- diagnose_full_smcs.py
|   `-- diagnose_quant_alignment.py
|-- scripts/
|   |-- local_stagewise_quant_eval.sh
|   |-- exp_quantdllm_ablation_quantize.sh
|   |-- exp_quantdllm_ablation_eval_mmlu.sh
|   |-- exp_gptq_llada8b_baseline.sh
|   `-- README_experiments.md
|-- docs/
|   |-- stagewise_gptq_daq_quantization.md
|   |-- reproduction_project_structure_beamer.tex
|   `-- reproduction_project_structure_beamer.pdf
|-- llada_benchmark.sh
`-- main.py
```

## Environment and Data

The scripts assume a Linux server environment with CUDA GPUs. The reproduction
used multi-GPU execution for LLaDA-8B-Base quantization and evaluation.

Expected inputs:

- LLaDA-8B-Base checkpoint, for example a local Hugging Face cache path.
- C4 calibration shard, for example
  `c4-train.00000-of-01024.json.gz`.
- A Python environment with PyTorch, Transformers, Accelerate, Datasets,
  Safetensors, and the evaluation dependencies used by `llada_benchmark.sh`.

The maintained launcher exposes most paths through environment variables:

- `PYTHON_BIN`
- `MODEL_PATH`
- `C4_SOURCE`
- `OUTPUT_DIR`
- `HF_HOME`
- `CUDA_VISIBLE_DEVICES`

## Reproduce the Mainline

Run from the repository root on the server:

```bash
RUN_ID=stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHON_BIN=/root/miniconda3/envs/qdlm/bin/python \
MODEL_PATH=/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base \
C4_SOURCE=/root/data-fs/c4-train/c4-train.00000-of-01024.json.gz \
NSAMPLES=128 \
SEQLEN=4096 \
NUM_STEPS=20 \
MAX_TOKENS_PER_STATE=0 \
DAMPING=0.0 \
DAMPING_MODE=diag_mean \
DAMP_PERCENT=0.01 \
GPU_MEMORY=13GiB \
CPU_MEMORY=120GiB \
DAQ_GRANULARITY=column \
ABMP_GRANULARITY=column \
ABMP_RANK_SCOPE=weight \
QUANT_BLOCK_SIZE=128 \
ALLOCATE_RATIO=0.05 \
DEFAULT_BITS=2 \
DAQ_STEPS=10 \
IMP_LAMDA=2.0 \
DOR_MASK_SCOPE=unit \
DAQ_COMPONENT=rsr_w_dor \
DAQ_DEVICE=cpu \
GPTQ_COMPENSATION=1 \
GPTQ_COMPENSATION_DEVICE=cuda:0 \
DAQ_FULLS_REFINE=0 \
RUN_EVAL=1 \
EVAL_TASKS=winogrande \
EVAL_LIMIT=full \
bash scripts/local_stagewise_quant_eval.sh
```

Use `EVAL_LIMIT=200` only for smoke tests. Do not compare limited evaluation
directly with paper tables.

## Results

### Full WinoGrande Check

For the current mainline:

```text
task: winogrande, 5-shot
samples: 1267 / 1267
acc: 0.6614048934490924
stderr: 0.013300169865842421
```

Reference points:

| Model / run | WinoGrande |
| --- | ---: |
| Local FP baseline | `0.7340` |
| Paper Quant-dLLM target | `0.6819` |
| Early reproduction quantized baseline | `0.6267` |
| Current mainline | `0.6614` |

### Common-Task Summary

The reproduced Quant-dLLM result below is averaged over the common tasks that
were fully evaluated in this run: WinoGrande, PIQA, ARC-C, ARC-E, and BBH.

| Method | MMLU(5) | Wino.(5) | PIQA(0) | ARC-C(0) | ARC-E(0) | Hella.(0) | BBH(0) | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP | - | 73.40 | 72.25 | 41.81 | 73.40 | - | 40.70 | 60.31 |
| GPTQ | 34.75 | 57.85 | 57.45 | 22.44 | 37.04 | 33.77 | 4.06 | 35.34 |
| GPTAQ | 34.73 | 59.04 | 56.42 | 22.27 | 38.17 | 35.13 | 5.34 | 35.87 |
| Slim-LLM | 47.98 | 60.85 | 62.46 | 26.11 | 53.41 | 39.25 | 6.64 | 42.39 |
| Quant-dLLM reproduction | - | 66.14 | 65.67 | 34.90 | 67.13 | - | 28.47 | 52.46 |

Compared with the paper's common-task Quant-dLLM average of `55.01`, the current
reproduction is `2.55` points lower. The local FP baseline is also lower than
the paper FP baseline by about `1.73` points on the same common tasks, leaving a
smaller residual quantization gap after accounting for evaluation-chain
differences.

## Diagnostic Variants

The repository keeps several non-mainline variants for analysis:

- `daq_fulls_refine`: fixed-binary-pattern full-S scale refinement.
- `daq_activation_diag`: diagonal activation-aware DAQ updates.
- `daq_row_center`, `daq_block_row_center`, `z_row_center`: row-centering
  ablations.
- `row`, `tile`, `weight` granularity: alternative DAQ/ABMP block layouts.
- full-SMCS Z cache builders and alignment diagnostics.

These variants are useful for debugging but are disabled in the verified
mainline unless explicitly enabled.

## Experiment Scripts

See [`scripts/README_experiments.md`](scripts/README_experiments.md) for:

- Quant-dLLM ablation quantization;
- MMLU ablation evaluation;
- GPTQ baseline runs;
- cross-model templates;
- layer-wise and stage-wise pilot commands;
- log monitoring patterns.

## Notes

- The repo is a reproduction project, not an official Quant-dLLM release.
- The strongest current result depends on GPTQ/OBC-style compensation for
  column groups.
- Smoke-test results with small `EVAL_LIMIT` are only for debugging.
- Large model forward/evaluation requires sufficient GPU memory and offload
  configuration.

## References

- Quant-dLLM paper: <https://arxiv.org/pdf/2510.03274>
- Slim-LLM paper: <https://arxiv.org/abs/2405.14917>
