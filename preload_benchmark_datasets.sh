#!/usr/bin/env bash
set -euo pipefail

export HF_DATASETS_TRUST_REMOTE_CODE=true
export HF_ALLOW_CODE_EVAL=1
export HF_HOME="${HF_HOME:-/root/data-fs/hf_shared}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"

QDLM_ROOT="${QDLM_ROOT:-/root/data-fs/QDLM}"
HARNESS_DIR="${HARNESS_DIR:-$QDLM_ROOT/lm-evaluation-harness}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_GROUP="${DATA_GROUP:-all}"

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "DATA_GROUP=$DATA_GROUP"
if [ -n "${http_proxy:-${HTTP_PROXY:-}}" ]; then
  echo "http_proxy=<set>"
else
  echo "http_proxy=<unset>"
fi
if [ -n "${https_proxy:-${HTTPS_PROXY:-}}" ]; then
  echo "https_proxy=<set>"
else
  echo "https_proxy=<unset>"
fi

"$PYTHON_BIN" - "$HARNESS_DIR" "$DATA_GROUP" <<'PY'
import os
import re
import sys
import time
from pathlib import Path

from datasets import load_dataset


HARNESS_DIR = Path(sys.argv[1])
DATA_GROUP = sys.argv[2]
TASK_ROOT = HARNESS_DIR / "lm_eval" / "tasks"


def yaml_scalar(path: Path, key: str):
    pattern = re.compile(rf"^\s*[\"']?{re.escape(key)}[\"']?\s*:\s*(.*?)\s*$")
    for line in path.read_text(errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        value = match.group(1).split("#", 1)[0].strip()
        value = value.strip('"').strip("'")
        if value in {"", "null", "None"}:
            return None
        return value
    return None


def dataset_names_from_yaml(directory: Path):
    names = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        name = yaml_scalar(path, "dataset_name")
        if name and name not in names:
            names.append(name)
    return names


def add_unique(items, path, name=None, splits=None):
    item = (path, name, tuple(splits or (None,)))
    if item not in items:
        items.append(item)


def build_items(group: str):
    items = []

    if group in {"paper_mc", "all"}:
        add_unique(items, "baber/piqa", splits=("train", "validation"))
        add_unique(items, "allenai/ai2_arc", "ARC-Easy", splits=("train", "validation", "test"))
        add_unique(items, "allenai/ai2_arc", "ARC-Challenge", splits=("train", "validation", "test"))
        add_unique(items, "hellaswag", splits=("train", "validation"))
        add_unique(items, "winogrande", "winogrande_xl", splits=("train", "validation"))

        for name in dataset_names_from_yaml(TASK_ROOT / "mmlu" / "default"):
            add_unique(items, "cais/mmlu", name, splits=("dev", "test"))

        for name in dataset_names_from_yaml(TASK_ROOT / "bbh" / "zeroshot"):
            add_unique(items, "SaylorTwift/bbh", name, splits=("test",))

    if group in {"reasoning", "all"}:
        add_unique(items, "gsm8k", "main", splits=("train", "test"))
        for name in dataset_names_from_yaml(TASK_ROOT / "minerva_math"):
            add_unique(items, "EleutherAI/hendrycks_math", name, splits=("train", "test"))

    if group in {"code", "all"}:
        add_unique(items, "openai/openai_humaneval", splits=("test",))
        add_unique(items, "google-research-datasets/mbpp", "full", splits=("test",))

    if group not in {"paper_mc", "reasoning", "code", "all"}:
        raise SystemExit(f"Unknown DATA_GROUP={group}. Use paper_mc, reasoning, code, or all.")

    return items


failures = []
items = build_items(DATA_GROUP)
print(f"Datasets to load: {len(items)}")
print(f"HF_HOME={os.environ.get('HF_HOME')}")
print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE')}")

for index, (path, name, splits) in enumerate(items, 1):
    label = path if name is None else f"{path}/{name}"
    split_text = ",".join(split for split in splits if split is not None) or "<all>"
    print(f"[{index:03d}/{len(items):03d}] loading {label} splits={split_text}", flush=True)

    for split in splits:
        split_label = label if split is None else f"{label}:{split}"
        started = time.time()
        try:
            if name is None:
                load_dataset(path, split=split, trust_remote_code=True)
            else:
                load_dataset(path, name, split=split, trust_remote_code=True)
            print(f"  OK {split_label} ({time.time() - started:.1f}s)", flush=True)
        except TypeError:
            try:
                if name is None:
                    load_dataset(path, split=split)
                else:
                    load_dataset(path, name, split=split)
                print(f"  OK {split_label} ({time.time() - started:.1f}s)", flush=True)
            except Exception as exc:
                failures.append((split_label, repr(exc)))
                print(f"  FAILED {split_label}: {exc}", flush=True)
        except Exception as exc:
            failures.append((split_label, repr(exc)))
            print(f"  FAILED {split_label}: {exc}", flush=True)

if failures:
    print("\nFailed datasets:")
    for label, error in failures:
        print(f"- {label}: {error}")
    raise SystemExit(1)

print("All requested datasets are available in the configured HF cache.")
PY
