#!/usr/bin/env python
# coding: utf-8
"""
split_ground_truth.py

Splits ground truth files into dev/val/test sets (60/20/20).
Uses shared indices derived from synthetic file to ensure
the same logical query lands in the same split across all three files.
Saves split_indices.json for reproducibility.
"""

import json
import random
from pathlib import Path

# -----------------------------
# Helper — latest file by prefix
# -----------------------------
def get_latest(gt_dir: Path, prefix: str) -> Path:
    files = sorted(
        gt_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not files:
        raise FileNotFoundError(f"No files found in {gt_dir} with prefix '{prefix}'")
    return files[0]

# -----------------------------
# Config
# -----------------------------
RANDOM_SEED = 42
DEV_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

GT_DIR = Path("data/eval/ground_truth")
OUTPUT_DIR = Path("data/eval")

FILES = {
    "synthetic":   get_latest(GT_DIR, "gt_gpt-5-nano_synthetic"),
    "paraphrased": get_latest(GT_DIR, "gt_gpt-5-nano_paraphrased"),
    "multi_doc":   get_latest(GT_DIR, "gt_llm_multi-doc"),
}

# Log which files were picked up
for name, path in FILES.items():
    print(f"[INFO] {name}: {path.name}")

# -----------------------------
# Load all three files
# -----------------------------
data = {}

for name, path in FILES.items():
    with open(path) as f:
        data[name] = json.load(f)
    print(f"[INFO] Loaded {len(data[name])} queries from {path.name}")

# Verify all three have same number of queries
lengths = [len(v) for v in data.values()]
assert len(set(lengths)) == 1, f"Query count mismatch across files: {lengths}"
n = lengths[0]
print(f"[INFO] All files have {n} queries — OK")

# -----------------------------
# Generate shared indices
# -----------------------------
random.seed(RANDOM_SEED)
indices = list(range(n))
random.shuffle(indices)

n_dev  = int(n * DEV_RATIO)
n_val  = int(n * VAL_RATIO)

dev_indices  = sorted(indices[:n_dev])
val_indices  = sorted(indices[n_dev:n_dev + n_val])
test_indices = sorted(indices[n_dev + n_val:])

print(f"\n[INFO] Split sizes:")
print(f"  dev:  {len(dev_indices)} queries ({len(dev_indices)/n*100:.1f}%)")
print(f"  val:  {len(val_indices)} queries ({len(val_indices)/n*100:.1f}%)")
print(f"  test: {len(test_indices)} queries ({len(test_indices)/n*100:.1f}%)")

# -----------------------------
# Save split indices for reproducibility
# -----------------------------
indices_path = OUTPUT_DIR / "split_indices.json"
with open(indices_path, "w") as f:
    json.dump({
        "seed": RANDOM_SEED,
        "n_total": n,
        "dev":  dev_indices,
        "val":  val_indices,
        "test": test_indices,
    }, f, indent=2)
print(f"\n[INFO] Saved split indices to {indices_path}")

# -----------------------------
# Write splits for each file
# -----------------------------
splits = {
    "dev":  dev_indices,
    "val":  val_indices,
    "test": test_indices,
}

for split_name, idx_list in splits.items():
    split_dir = OUTPUT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for query_type, queries in data.items():
        split_queries = [queries[i] for i in idx_list]
        out_path = split_dir / f"{query_type}.json"
        with open(out_path, "w") as f:
            json.dump(split_queries, f, indent=2)
        print(f"[OK] {split_name}/{query_type}.json — {len(split_queries)} queries")

# -----------------------------
# Also copy originals to all/
# -----------------------------
all_dir = OUTPUT_DIR / "all"
all_dir.mkdir(parents=True, exist_ok=True)

for query_type, queries in data.items():
    out_path = all_dir / f"{query_type}.json"
    with open(out_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"[OK] all/{query_type}.json — {len(queries)} queries")

print(f"\n[DONE] Split complete. Structure:")
print(f"  data/eval/all/       ← full unsplit files")
print(f"  data/eval/dev/       ← {len(dev_indices)} queries per type")
print(f"  data/eval/val/       ← {len(val_indices)} queries per type")
print(f"  data/eval/test/      ← {len(test_indices)} queries per type  🔒")
print(f"  data/eval/split_indices.json  ← reproducibility record")