import argparse
import gzip
import json
import random
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer

MASK_TOKEN = "<|mdm_mask|>"
MODEL_PATH = "/root/data-fs/.cache/hub/models--GSAI-ML--LLaDA-8B-Base/snapshots/LLaDA-8B-Base"
C4_TRAIN_URL = "https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz"


def build_tokenizer(model_path=MODEL_PATH, local_files_only=False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    mask_token_id = tokenizer.convert_tokens_to_ids(MASK_TOKEN)
    if mask_token_id is None:
        raise ValueError(f"Mask token {MASK_TOKEN!r} is not in tokenizer vocab.")
    return tokenizer, mask_token_id


def linear_mask(t, num_steps):
    return t / num_steps


def cosine_mask(t, num_steps):
    return 0.5 * (1 - torch.cos(torch.tensor(t / num_steps) * torch.pi)).item()


@contextmanager
def open_gzip_jsonl(source):
    if source.startswith("http://") or source.startswith("https://"):
        request = urllib.request.Request(source, headers={"User-Agent": "quant-dllm-mcs"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with gzip.GzipFile(fileobj=response) as gzip_file:
                yield gzip_file
    else:
        with gzip.open(Path(source), "rb") as gzip_file:
            yield gzip_file


def iter_c4_text(source=C4_TRAIN_URL):
    with open_gzip_jsonl(source) as gzip_file:
        for line in gzip_file:
            if not line:
                continue
            yield json.loads(line)["text"]


def get_c4(nsamples, seed, seqlen, source=C4_TRAIN_URL, model_path=MODEL_PATH, max_docs=None):
    tokenizer, _ = build_tokenizer(model_path=model_path, local_files_only=Path(model_path).exists())
    rng = random.Random(seed)
    trainloader = []

    for doc_idx, text in enumerate(iter_c4_text(source)):
        if max_docs is not None and doc_idx >= max_docs:
            break

        input_ids = tokenizer(text, return_tensors="pt").input_ids
        token_count = input_ids.shape[1]
        if token_count < seqlen:
            continue

        start = rng.randint(0, token_count - seqlen)
        trainloader.append(input_ids[:, start : start + seqlen])
        if len(trainloader) >= nsamples:
            break

    if len(trainloader) < nsamples:
        raise RuntimeError(
            f"Only collected {len(trainloader)} C4 samples with at least {seqlen} tokens. "
            f"Try increasing --max-docs or using the full C4 shard."
        )

    return trainloader


def MCS(dataloader, num_steps, gama, maskratio_func, mask_token_id):
    calibset = []

    for sentence in dataloader:
        data = sentence.squeeze(0)
        length = data.numel()
        prefix_len = int(gama * length)

        for t in range(num_steps + 1):
            visible_prob = float(maskratio_func(t, num_steps))

            masked_data = data.clone()
            if prefix_len < length:
                visible = torch.bernoulli(
                    torch.full((length - prefix_len,), visible_prob, dtype=torch.float32)
                ).bool()
                masked_data[prefix_len:][~visible] = mask_token_id

            calibset.append(masked_data.unsqueeze(0))
    return calibset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--source", default=C4_TRAIN_URL, help="C4 .json.gz URL or local path.")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=0.25)
    parser.add_argument("--schedule", choices=["linear", "cosine"], default="linear")
    parser.add_argument("--max-docs", type=int, default=None)
    return parser.parse_args()


def get_calib():
    args = parse_args()
    _, mask_token_id = build_tokenizer(
        model_path=args.model,
        local_files_only=Path(args.model).exists(),
    )
    schedule = linear_mask if args.schedule == "linear" else cosine_mask

    trainloader = get_c4(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        source=args.source,
        model_path=args.model,
        max_docs=args.max_docs,
    )
    calibset = MCS(trainloader, args.num_steps, args.gamma, schedule, mask_token_id)

    return args, trainloader, calibset
    # calibset: List[tokens1,tokens2,...]
    
def main():
    args, trainloader, calibset = get_calib()
    print("c4_samples:", len(trainloader))
    print("mcs_samples:", len(calibset))
    print("sample_shape:", tuple(calibset[0].shape))
    for i in range(min(args.num_steps + 1, len(calibset))):
        print(i, calibset[i][:, :50])


if __name__ == "__main__":
    main()
