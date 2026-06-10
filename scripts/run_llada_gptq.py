#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
QDLM_ROOT = Path(os.environ.get("QDLM_ROOT", "/root/data-fs/QDLM"))
AUTOGPTQ_ROOT = QDLM_ROOT / "AutoGPTQ"
HARNESS_ROOT = QDLM_ROOT / "lm-evaluation-harness"

for path in (str(REPO_ROOT), str(AUTOGPTQ_ROOT), str(HARNESS_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize/evaluate LLaDA with QDLM AutoGPTQ.")
    parser.add_argument("--model_path", default="/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base")
    parser.add_argument("--quantized_model_dir", required=True)
    parser.add_argument("--results_dir", default="/root/data-fs/Quant-dllm/benchmark_results/gptq_llada8b")
    parser.add_argument("--tasks", default="mmlu")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--wbits", type=int, default=2)
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--desc_act", action="store_true")
    parser.add_argument("--calib_source", default="https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz")
    parser.add_argument("--calib_nsamples", type=int, default=128)
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--calib_seqlen", type=int, default=4096)
    parser.add_argument("--calib_max_docs", type=int, default=None)
    parser.add_argument("--mc_num", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use_triton", action="store_true")
    parser.add_argument("--quantize_only", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--output_name", default=None)
    return parser.parse_args()


def patch_llada_autogptq_layer_names():
    from auto_gptq.modeling.llada import LladaGPTQForCausalLM

    LladaGPTQForCausalLM.inside_layer_modules = [
        ["attn_out"],
        ["q_proj", "k_proj", "v_proj"],
        ["ff_out"],
        ["up_proj", "ff_proj"],
    ]


def build_c4_examples(args):
    from MCS.load_dataset import get_c4

    samples = get_c4(
        nsamples=args.calib_nsamples,
        seed=args.calib_seed,
        seqlen=args.calib_seqlen,
        source=args.calib_source,
        model_path=args.model_path,
        max_docs=args.calib_max_docs,
    )
    return [
        {
            "input_ids": sample,
            "attention_mask": torch.ones_like(sample),
        }
        for sample in samples
    ]


def load_or_quantize_model(args):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    patch_llada_autogptq_layer_names()

    quantized_model_dir = Path(args.quantized_model_dir)
    if quantized_model_dir.exists() and any(quantized_model_dir.iterdir()):
        print(f"Loading existing GPTQ model from {quantized_model_dir}", flush=True)
        return AutoGPTQForCausalLM.from_quantized(
            str(quantized_model_dir),
            device=args.device,
            use_triton=args.use_triton,
            trust_remote_code=True,
        )

    if args.eval_only:
        raise FileNotFoundError(f"--eval_only was set but GPTQ model does not exist: {quantized_model_dir}")

    quantize_config = BaseQuantizeConfig(
        bits=args.wbits,
        group_size=args.group_size,
        desc_act=args.desc_act,
    )

    print("Loading base model for GPTQ quantization", flush=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_path,
        quantize_config,
        trust_remote_code=True,
    )
    print("Building C4 calibration examples", flush=True)
    examples = build_c4_examples(args)
    print(f"Quantizing with {len(examples)} examples, seqlen={args.calib_seqlen}", flush=True)
    model.quantize(examples, use_triton=args.use_triton)
    quantized_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(quantized_model_dir), use_safetensors=True)
    print(f"Saved GPTQ model to {quantized_model_dir}", flush=True)
    return model


def evaluate(args, model):
    from lm_eval import evaluator
    from lm_eval.api.registry import get_model

    model = model.to(args.device)
    model.eval()

    model_cls = get_model("llada_dist")
    model_args = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "mc_num": args.mc_num,
        "batch_size": args.batch_size,
        "is_check_greedy": False,
        "dtype": args.dtype,
    }
    lm = model_cls(model=model, model_path=args.model_path, **model_args)

    with torch.cuda.amp.autocast(enabled=args.dtype in {"float16", "fp16", "bfloat16", "bf16"}):
        results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks.split(","),
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
            model_args=model_args,
            confirm_run_unsafe_code=True,
        )

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or args.tasks.replace(",", "_")
    output_path = results_dir / f"{output_name}.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {output_path}", flush=True)


def main():
    args = parse_args()
    model = load_or_quantize_model(args)
    if args.quantize_only:
        return
    evaluate(args, model)


if __name__ == "__main__":
    main()
