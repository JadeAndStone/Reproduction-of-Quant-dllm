import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


MASK_TOKEN_CANDIDATES = (
    "<|mdm_mask|>",
    "<|mask|>",
    "[MASK]",
    "<mask>",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Load a dLLM checkpoint and report whether it matches the current "
            "Quant-dLLM/LLaDA reproduction assumptions."
        )
    )
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--auto-class", choices=["auto", "causal"], default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--gpu-memory", default="13GiB")
    parser.add_argument("--cpu-memory", default="80GiB")
    parser.add_argument("--offload-folder", default="/root/data-tmp/offload/probe_dllm_model")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-model-load", action="store_true")
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument(
        "--attention-mask-mode",
        choices=["auto", "keep", "bool", "drop"],
        default="auto",
        help="How to handle tokenizer attention_mask during forward smoke. Dream remote code needs bool or no mask.",
    )
    parser.add_argument("--prompt", default="The answer is")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args()


def dtype_from_name(name):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def build_max_memory(args):
    if torch.cuda.is_available():
        max_memory = {gpu_idx: args.gpu_memory for gpu_idx in range(torch.cuda.device_count())}
        max_memory["cpu"] = args.cpu_memory
        return max_memory
    return {"cpu": args.cpu_memory}


def tensor_device_from_module(module):
    for tensor in list(module.parameters(recurse=False)) + list(module.buffers(recurse=False)):
        if tensor.device.type != "meta":
            return tensor.device
    return None


def first_tensor_device(model):
    for module_path in (
        "model.transformer.wte",
        "transformer.wte",
        "model.embed_tokens",
        "model.embeddings.word_embeddings",
        "model.tok_embeddings",
    ):
        module = model
        for attr in module_path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is None:
            continue
        device = tensor_device_from_module(module)
        if device is not None:
            return device
    for tensor in model.parameters():
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resolve_attr(root, path):
    obj = root
    for attr in path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def get_len(obj):
    try:
        return len(obj)
    except TypeError:
        return None


def summarize_blocks(model):
    candidates = (
        "model.transformer.blocks",
        "transformer.blocks",
        "model.layers",
        "layers",
        "model.decoder.layers",
        "backbone.layers",
    )
    found = []
    for path in candidates:
        blocks = resolve_attr(model, path)
        if blocks is None:
            continue
        length = get_len(blocks)
        first = blocks[0] if length else None
        first_attrs = []
        if first is not None:
            for attr in (
                "q_proj",
                "k_proj",
                "v_proj",
                "attn_out",
                "o_proj",
                "ff_proj",
                "up_proj",
                "ff_out",
                "gate_proj",
                "down_proj",
            ):
                if hasattr(first, attr):
                    first_attrs.append(attr)
        found.append(
            {
                "path": path,
                "num_blocks": length,
                "first_block_class": type(first).__name__ if first is not None else None,
                "first_block_direct_attrs": first_attrs,
            }
        )
    return found


def summarize_module_names(model):
    suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "attn_out",
        "o_proj",
        "ff_proj",
        "up_proj",
        "ff_out",
        "gate_proj",
        "down_proj",
    )
    counts = {suffix: 0 for suffix in suffixes}
    examples = {suffix: [] for suffix in suffixes}
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        short = name.rsplit(".", 1)[-1]
        if short in counts:
            counts[short] += 1
            if len(examples[short]) < 3:
                examples[short].append(name)
    return {"linear_suffix_counts": counts, "linear_suffix_examples": examples}


def quant_dllm_compatibility(blocks_summary, module_summary):
    counts = module_summary["linear_suffix_counts"]
    has_llada_blocks = any(item["path"] == "model.transformer.blocks" for item in blocks_summary)
    has_dream_blocks = any(item["path"] == "model.layers" for item in blocks_summary)
    llada_required = ("q_proj", "k_proj", "v_proj", "attn_out", "ff_proj", "up_proj", "ff_out")
    dream_required = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    has_llada_required = all(counts.get(name, 0) > 0 for name in llada_required)
    has_dream_required = all(counts.get(name, 0) > 0 for name in dream_required)
    supported = (has_llada_blocks and has_llada_required) or (has_dream_blocks and has_dream_required)
    if has_llada_blocks and has_llada_required:
        reason = "matches LLaDA model.transformer.blocks projection layout"
    elif has_dream_blocks and has_dream_required:
        reason = "matches Dream model.layers self_attn/MLP projection layout"
    else:
        reason = "model block layout does not match the supported LLaDA or Dream projection sets"
    return {"stagewise_likely_supported": bool(supported), "reason": reason}


def main():
    args = parse_args()
    torch_dtype = dtype_from_name(args.dtype)
    result = {"model": args.model}

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    mask_candidates = {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in MASK_TOKEN_CANDIDATES
    }
    result["tokenizer"] = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "mask_token": getattr(tokenizer, "mask_token", None),
        "mask_token_id": getattr(tokenizer, "mask_token_id", None),
        "mask_candidates": mask_candidates,
    }

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    result["config"] = {
        "class": type(config).__name__,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "vocab_size": getattr(config, "vocab_size", None),
    }

    if args.skip_model_load:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.json_out:
            Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        return

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map and args.device_map.lower() not in ("none", "false", ""):
        Path(args.offload_folder).mkdir(parents=True, exist_ok=True)
        model_kwargs.update(
            {
                "device_map": args.device_map,
                "max_memory": build_max_memory(args),
                "offload_folder": args.offload_folder,
                "offload_state_dict": True,
            }
        )

    model_cls = AutoModel if args.auto_class == "auto" else AutoModelForCausalLM
    model = model_cls.from_pretrained(args.model, **model_kwargs)
    model.eval()

    blocks_summary = summarize_blocks(model)
    module_summary = summarize_module_names(model)
    result["model_loaded"] = {
        "class": type(model).__name__,
        "device_map": getattr(model, "hf_device_map", None),
        "blocks": blocks_summary,
        **module_summary,
    }
    result["quant_dllm_compatibility"] = quant_dllm_compatibility(blocks_summary, module_summary)

    if not args.skip_forward:
        encoded = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=True)
        attention_mask_mode = args.attention_mask_mode
        model_type = str(getattr(config, "model_type", "") or "").lower()
        if attention_mask_mode == "auto" and model_type == "dream":
            attention_mask_mode = "drop"
        if "attention_mask" in encoded:
            if attention_mask_mode == "drop":
                encoded.pop("attention_mask", None)
            elif attention_mask_mode == "bool":
                encoded["attention_mask"] = encoded["attention_mask"].bool()
        input_device = first_tensor_device(model)
        encoded = {key: value.to(input_device) for key, value in encoded.items()}
        with torch.no_grad():
            try:
                output = model(**encoded, use_cache=False)
            except TypeError:
                output = model(**encoded)
        logits = getattr(output, "logits", None)
        result["forward_smoke"] = {
            "input_device": str(input_device),
            "input_shape": list(encoded["input_ids"].shape),
            "attention_mask_mode": attention_mask_mode,
            "encoded_keys": sorted(encoded.keys()),
            "has_logits": logits is not None,
            "logits_shape": list(logits.shape) if logits is not None else None,
            "logits_dtype": str(logits.dtype) if logits is not None else None,
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
