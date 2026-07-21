# Stage-wise GPTQ-compensated DAQ Quantization

This document describes the current reproduction mainline used for the latest LLaDA-8B-Base quantization result:

```text
stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_20260615_023437
```

The key enabled method is paper-style DAQ/DOR with column-group ABMP plus GPTQ/OBC-style error compensation. The fixed-B full-S scale refinement code exists as an optional diagnostic/enhanced variant, but it is not enabled in this mainline configuration.

## 1. Configuration

The verified mainline configuration is:

| Item | Value |
| --- | --- |
| Calibration samples | `nsamples=128` |
| Sequence length | `seqlen=4096` |
| MCS steps | `num_steps=20` |
| Mask schedule | `linear` |
| Fully visible endpoint | disabled by `--mcs_no_full_visible` |
| MCS normalization | `tokens` |
| Token subsampling per state | `max_tokens_per_state=0` meaning use all tokens |
| SMCS damping | `damping_mode=diag_mean`, `damp_percent=0.01`, `damping=0.0` |
| Quantization unit | column group, `--daq_granularity column` |
| ABMP unit | column group, `--abmp_granularity column` |
| ABMP rank scope | per weight, `--abmp_rank_scope weight` |
| Group size | `quant_block_size=128` input channels |
| Base bit-width | `default_bits=2` |
| Mixed precision ratio | `allocate_ratio=0.05` |
| DAQ component | `rsr_w_dor` |
| DAQ steps | `daq_steps=10` |
| DOR weight | `imp_lamda=2.0` |
| DOR mask scope | per DAQ unit, `dor_mask_scope=unit` |
| GPTQ compensation | enabled |
| Row centering | disabled for `W` and `Z` |
| Save dtype | `fp16` model weights, `fp32` Z cache |

The corresponding launcher is `scripts/run_quantization.sh`; pass the above values through environment variables, especially `GPTQ_COMPENSATION=1`, `DAQ_GRANULARITY=column`, `ABMP_GRANULARITY=column`, `ABMP_RANK_SCOPE=weight`, `MAX_TOKENS_PER_STATE=0`, `DAMPING_MODE=diag_mean`, and `DAMP_PERCENT=0.01`.

## 2. Tensor Shapes and Objective

For each target `Linear` layer, the model computes:

```text
Y = X W^T
```

where:

- `W in R^{d_out x d_in}` is the original weight.
- `X in R^{N x d_in}` is the collected activation input to this `Linear`, flattened over calibration states and tokens.
- `Y in R^{N x d_out}` is the corresponding output contribution.

The output-aware reconstruction objective is:

```text
||X W^T - X Q^T||_F^2
  = Tr((W - Q) S (W - Q)^T)
```

with the uncentered second moment / SMCS:

```text
S = X^T X / N
```

In code, hooks accumulate `X^T X` for each target `Linear`; `N` is `token_counts[weight_name]` when `mcs_normalization=tokens`.

## 3. Stage-wise Pipeline

The current mainline uses `tools/quant_stagewise_pipeline.py`. It processes each transformer block sequentially and, inside a block, quantizes the following stages in order:

1. `qkv`: `q_proj`, `k_proj`, `v_proj`
2. `attn_out`: `attn_out`
3. `ff_in`: `ff_proj`, `up_proj`
4. `ff_out`: `ff_out`

For each block:

1. Build or load hidden states entering the block.
2. For one stage, register hooks only on that stage's target `Linear` modules.
3. Run the stage forward to collect that stage's current `X^T X` statistics.
4. Build `Z` and precision allocation for the stage.
5. Quantize the stage and immediately write the quantized weights back into the in-memory model.
6. Continue to the next stage, so later stage statistics see earlier quantized modules in the same block.
7. After all stages in the block are quantized, propagate the full quantized block output as the hidden cache for the next block.

This stage-wise behavior is intentionally different from an offline Z-cache pipeline: statistics are generated from the current model state instead of reusing a fixed Z cache across incompatible variants.

## 4. SMCS and Z Construction

For each target weight, the accumulated covariance is normalized first:

```text
S = cov / normalizer
```

With the verified configuration, `normalizer` is the number of observed activation tokens for that weight.

A damping value is then selected:

```text
epsilon = damp_percent * mean(diag(S))       if damping_mode = diag_mean
```

The damped inverse is:

```text
H = (S + epsilon I)^(-1)
```

The implementation checks that `diag(H)` is finite and positive. If the check fails, damping is multiplied by 10 and retried, up to the configured retry limit.

Quant-dLLM's distilled importance matrix is constructed from the inverse diagonal:

```text
D = diag(H)
Z_ij = (W_ij / max(D_j, inverse_diag_floor))^2
```

Equivalently, each input-channel column `j` is scaled by the inverse diagonal value derived from the full SMCS for that `Linear` layer. In this mainline, no row-centering is applied to `W` before constructing `Z`.

The implementation also keeps an upper Cholesky factor of `H` for GPTQ compensation:

```text
H = U^T U
```

where `U` is the upper-triangular factor used during column-group error propagation.

## 5. ABMP Precision Allocation

The current ABMP unit is a full-row column group:

```text
G_k = W[:, 128k : 128(k+1)]
```

For the matching Z slice:

```text
score(G_k) = sum_{i,j in G_k} Z_ij
```

Because `abmp_rank_scope=weight`, groups are ranked only against other groups from the same `Linear` weight. With `allocate_ratio=0.05`:

- lowest 5% score groups receive 1 bit;
- middle groups receive 2 bits;
- highest 5% score groups receive 3 bits.

This keeps the average precision near 2 bits per weight while assigning more precision to high-Z column groups.

## 6. DOR Mask and DAQ Objective

For each DAQ unit, DOR builds a 3-sigma mask from that unit's `Z` slice:

```text
Pi_ij = 1[ |Z_ij - mean(Z_unit)| > 3 * std(Z_unit) ]
Lambda_ij = 1 + (lambda - 1) Pi_ij
```

With `imp_lamda=2.0`, selected outlier positions get twice the reconstruction weight.

DAQ optimizes the weighted proxy objective:

```text
min_Q || Lambda o (W_unit - Q) ||_F^2
```

where `o` is elementwise multiplication.

For `b` bits, DAQ represents the quantized unit as a sum of binary rank-1 components:

```text
Q = sum_{t=1}^{b} (alpha_r^{(t)} (alpha_c^{(t)})^T) o B^{(t)}
```

where:

- `alpha_r^{(t)} in R^{rows x 1}` is a row scale vector.
- `alpha_c^{(t)} in R^{cols x 1}` is a column scale vector.
- `B^{(t)} in {-1,+1}^{rows x cols}` is the binary sign matrix for component `t`.

For each component, after removing the contribution of all other components:

```text
R^{(t)} = W_unit - sum_{s != t} (alpha_r^{(s)} (alpha_c^{(s)})^T) o B^{(s)}
```

DAQ alternates closed-form scale updates and binary pattern updates.

The scale updates used by the DOR-weighted objective are:

```text
alpha_r = ((Lambda^2 o R o B) alpha_c) / ((Lambda^2) (alpha_c^2) + eps)
alpha_c = ((Lambda^2 o R o B)^T alpha_r) / ((Lambda^2)^T (alpha_r^2) + eps)
```

After scale updates, `B` is selected elementwise by evaluating all `2^b` bit patterns and choosing the pattern with the smallest squared error to `W_unit` under the current rank-1 scale components.

## 7. GPTQ/OBC-style Compensation

The mainline enables GPTQ-style compensation for column DAQ. This requires column groups that cover all output rows:

```text
W[:, c_start:c_end]
```

For each column group in order:

1. Quantize the current working slice `W_work[:, c_start:c_end]` with DAQ/DOR.
2. Save the quantized result into `Q`.
3. Propagate the quantization error to later columns using the inverse factor `U`.

The implemented update is:

```text
E = (W_work[:, c_start:c_end] - Q[:, c_start:c_end]) / diag(U_block)
W_work[:, c_end:] = W_work[:, c_end:] - E U_cross
```

where:

- `U_block = U[c_start:c_end, c_start:c_end]`
- `U_cross = U[c_start:c_end, c_end:]`

This follows the GPTQ/OBC idea: once a group is quantized, the remaining unquantized columns are adjusted so that future quantization sees a compensated residual. In the current reproduction, this is the most important implementation difference between the early baseline and the stronger mainline.

## 8. Optional Non-mainline Variants

The code also contains diagnostic variants. They are intentionally not part of the mainline result unless their flags are explicitly enabled:

- `--daq_fulls_refine`: fixed-B full-S scale refinement. It refines `alpha_r` and `alpha_c` using the local Hessian/covariance submatrix while keeping `B` fixed. This is an engineering diagnostic/enhancement, not the paper-literal DAQ/DOR path.
- `--daq_activation_diag`: diagonal activation-aware DAQ updates. This minimizes a diagonal-Hessian weighted proxy, not the original DAQ/DOR proxy.
- `--daq_row_center`, `--daq_block_row_center`, `--z_row_center`: row-centering ablations. These are disabled in the verified mainline.
- `row`, `tile`, or `weight` granularity: useful ablations, but the verified mainline uses column groups.

## 9. Reproduction Command

A local run can be launched through the maintained wrapper:

```bash
cd /root/data-fs/Quant-dllm-repro

RUN_ID=stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_$(date +%Y%m%d_%H%M%S) \
CUDA_VISIBLE_DEVICES=0,1,2 \
PYTHON_BIN=/root/miniconda3/envs/qdlm/bin/python \
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
bash scripts/run_quantization.sh
```

`EVAL_LIMIT=full` runs the full benchmark. Use `EVAL_LIMIT=200` only for smoke tests and do not compare it directly against paper tables.

## 10. Latest Verified WinoGrande Result

For the A mainline model:

```text
model: outputs/stagewise_gptq_comp_colabmp_n128_s4096_t20_tok0_diag1p_20260615_023437
task: winogrande, 5-shot
limit: None
samples: 1267 / 1267
acc: 0.6614048934490924
stderr: 0.013300169865842421
```

For comparison:

- FP local baseline: `0.7340`
- paper Quant-dLLM target: `0.6819`
- early reproduction quantized baseline: `0.6267`
- current A mainline: `0.6614`

A separate GPTQ-plus-full-S variant reached `0.6622` on full WinoGrande, which is statistically indistinguishable from the A mainline. That result indicates fixed-B full-S refinement is not needed to explain the WinoGrande improvement.
