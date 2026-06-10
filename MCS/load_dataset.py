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

def build_tokenizer(model_path, local_files_only=False):
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


def linear_mask(tau):
    return 1.0 - tau


def cosine_mask(tau):
    return 0.5 * (1 + torch.cos(torch.tensor(tau) * torch.pi)).item()


def build_timestep_grid(num_steps, include_full_visible=True):
    if num_steps <= 0:
        raise ValueError("MCS needs at least one mask state. Use --num-steps > 0.")

    if include_full_visible:
        if num_steps == 1:
            return [0.0]
        return torch.linspace(0.0, 1.0, num_steps, dtype=torch.float32).tolist()

    return torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.float32)[1:].tolist()


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


def iter_c4_text(source):
    with open_gzip_jsonl(source) as gzip_file:
        for line in gzip_file:
            if not line:
                continue
            yield json.loads(line)["text"]


def get_c4(nsamples, seed, seqlen, source, model_path, max_docs):
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


def MCS(
    dataloader,
    num_steps,
    gama,
    maskratio_func,
    mask_token_id,
    include_full_visible=True,
    seed=None,
):
    calibset = []
    timestep_grid = build_timestep_grid(
        num_steps,
        include_full_visible=include_full_visible,
    )
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    for sentence in dataloader:
        data = sentence.squeeze(0)
        length = data.numel()
        prefix_len = int(gama * length)

        for tau in timestep_grid:
            visible_prob = float(maskratio_func(tau))

            masked_data = data.clone()
            if prefix_len < length:
                visible = torch.bernoulli(
                    torch.full((length - prefix_len,), visible_prob, dtype=torch.float32),
                    generator=generator,
                ).bool()
                masked_data[prefix_len:][~visible] = mask_token_id

            calibset.append(masked_data.unsqueeze(0))
    return calibset, len(timestep_grid)



def get_calib(model_path,nsamples,
              seed,seqlen,
              num_steps,gamma,
              schedule,
              max_docs,source,
              include_full_visible=True):

    _, mask_token_id = build_tokenizer(
        model_path=model_path,
        local_files_only=Path(model_path).exists(),
    )
    schedule = linear_mask if schedule == "linear" else cosine_mask

    trainloader = get_c4(
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        source=source,
        model_path=model_path,
        max_docs=max_docs,
    )
    calibset, mcs_step_count = MCS(
        trainloader,
        num_steps,
        gamma,
        schedule,
        mask_token_id,
        include_full_visible=include_full_visible,
        seed=seed,
    )

    return trainloader, calibset, mcs_step_count
    # calibset: List[tokens1,tokens2,...]
