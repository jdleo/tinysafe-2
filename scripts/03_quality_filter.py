#!/usr/bin/env python3
"""
Step 3: Filter, deduplicate, balance, and split the v2 dataset.
- Contamination check against all eval benchmarks
- Exact + near dedup
- Class balance
- Stratified split
"""

import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from src.utils import CATEGORIES, load_config, load_jsonl, save_jsonl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def _label_counts(samples: list[dict]) -> str:
    safe = sum(1 for s in samples if s.get("label") == "safe")
    unsafe = len(samples) - safe
    return f"[safe={safe}, unsafe={unsafe}]"


def build_eval_hashes() -> set[str]:
    from datasets import load_dataset

    eval_hashes = set()
    eval_texts = []

    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
        for row in ds:
            text = row.get("user_input", "")
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load ToxicChat test: {e}")

    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        for row in ds:
            text = row.get("prompt", "") or row.get("instruction", "") or ""
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load WildGuardBench test: {e}")

    try:
        ds = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
        for row in ds:
            text = row.get("prompt", "")
            if text.strip():
                eval_texts.append(text)
    except Exception as e:
        print(f"  Warning: could not load OR-Bench: {e}")

    for text in eval_texts:
        eval_hashes.add(hashlib.sha256(text.strip().lower().encode()).hexdigest())

    print(f"  Built eval hash set: {len(eval_hashes)} unique eval texts")
    return eval_hashes


def contamination_filter(samples: list[dict], eval_hashes: set[str]) -> list[dict]:
    before = len(samples)
    filtered = []
    contaminated = 0
    for s in samples:
        h = hashlib.sha256(s["text"].strip().lower().encode()).hexdigest()
        if h in eval_hashes:
            contaminated += 1
        else:
            filtered.append(s)
    print(f"  Contamination filter: {before} -> {len(filtered)} (removed {contaminated}) {_label_counts(filtered)}")
    return filtered


def exact_dedup(samples: list[dict]) -> list[dict]:
    before = len(samples)
    seen = set()
    deduped = []
    for s in samples:
        h = hashlib.sha256(s["text"].strip().lower().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(s)
    print(f"  Exact dedup: {before} -> {len(deduped)} (dropped {before - len(deduped)}) {_label_counts(deduped)}")
    return deduped


def near_dedup(samples: list[dict], threshold: float = 0.95) -> list[dict]:
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("  Near dedup: skipped (datasketch not installed)")
        return samples

    before = len(samples)
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    minhashes = []

    for i, s in enumerate(tqdm(samples, desc="  Computing MinHashes")):
        m = MinHash(num_perm=128)
        words = s["text"].strip().lower().split()
        for w in words:
            m.update(w.encode("utf-8"))
        minhashes.append(m)
        try:
            lsh.insert(f"s-{i}", m)
        except ValueError:
            pass

    to_remove = set()
    for i, m in enumerate(minhashes):
        if i in to_remove:
            continue
        neighbors = lsh.query(m)
        for n in neighbors:
            idx = int(n.split("-")[1])
            if idx != i and idx not in to_remove:
                to_remove.add(idx)

    deduped = [s for i, s in enumerate(samples) if i not in to_remove]
    print(f"  Near dedup (>{threshold}): {before} -> {len(deduped)} (dropped {before - len(deduped)}) {_label_counts(deduped)}")
    return deduped


def length_filter(samples: list[dict], min_tokens: int, max_tokens: int) -> list[dict]:
    effective_min = min(min_tokens, 3)
    before = len(samples)
    filtered = [s for s in samples if effective_min <= len(s["text"].split()) <= max_tokens]
    print(f"  Length filter ({effective_min}-{max_tokens}): {before} -> {len(filtered)} (dropped {before - len(filtered)}) {_label_counts(filtered)}")
    return filtered


def class_balance(samples: list[dict], safe_ratio: float, unsafe_ratio: float) -> list[dict]:
    random.seed(42)
    safe = [s for s in samples if s["label"] == "safe"]
    unsafe = [s for s in samples if s["label"] == "unsafe"]

    if len(unsafe) == 0 or len(safe) == 0:
        return samples

    target_safe = int(len(unsafe) * safe_ratio / unsafe_ratio)

    if len(safe) > target_safe:
        safe = random.sample(safe, target_safe)
        print(f"  Class balance: downsampled safe to {len(safe)}")
    else:
        print(f"  Class balance: keeping all {len(safe)} safe samples")

    balanced = safe + unsafe
    random.shuffle(balanced)
    print(f"  Class balance: {len(safe)} safe ({len(safe)/len(balanced)*100:.1f}%), "
          f"{len(unsafe)} unsafe ({len(unsafe)/len(balanced)*100:.1f}%)")
    return balanced


def stratified_split(samples: list[dict], train_ratio: float, val_ratio: float, test_ratio: float):
    random.seed(42)
    groups = {}
    for s in samples:
        key = (s.get("source", "unknown"), s["label"])
        groups.setdefault(key, []).append(s)

    train, val, test = [], [], []
    for key, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])

    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    return train, val, test


def main():
    config = load_config()
    filter_config = config["filtering"]
    split_config = config["splits"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    samples = []

    # 1. Base data (ToxicChat + Jigsaw)
    base_path = RAW_DIR / "all_base_data.jsonl"
    if base_path.exists():
        data = load_jsonl(base_path)
        print(f"Loaded {len(data)} from all_base_data.jsonl")
        samples.extend(data)
    else:
        # Load individual files
        for name in ["toxicchat_train.jsonl", "jigsaw_ub.jsonl", "jigsaw_tc.jsonl"]:
            path = RAW_DIR / name
            if path.exists():
                data = load_jsonl(path)
                print(f"Loaded {len(data)} from {name}")
                samples.extend(data)

    # 2. Hard negatives
    hn_path = RAW_DIR / "hard_negatives.jsonl"
    if hn_path.exists():
        data = load_jsonl(hn_path)
        print(f"Loaded {len(data)} hard negatives")
        samples.extend(data)

    # 3. Synthetic data for missing categories (self_harm, dangerous_info, illegal_activity)
    syn_path = RAW_DIR / "synthetic_missing_categories.jsonl"
    if syn_path.exists():
        data = load_jsonl(syn_path)
        print(f"Loaded {len(data)} synthetic missing-category samples")
        samples.extend(data)

    # 4. WildGuard train (adversarial/jailbreak prompts)
    wg_path = RAW_DIR / "wildguard_train.jsonl"
    if wg_path.exists():
        data = load_jsonl(wg_path)
        print(f"Loaded {len(data)} WildGuard train samples")
        samples.extend(data)

    if not samples:
        print("ERROR: No data found. Run 01_prepare_data.py first.")
        sys.exit(1)

    print(f"\nTotal loaded: {len(samples)} {_label_counts(samples)}")

    # Build eval hash set
    print("\nBuilding eval benchmark hash set...")
    eval_hashes = build_eval_hashes()

    # Filters
    print("\nApplying filters:")
    samples = contamination_filter(samples, eval_hashes)
    samples = exact_dedup(samples)
    samples = near_dedup(samples, filter_config["dedup_similarity_threshold"])
    samples = length_filter(samples, filter_config["min_tokens"], filter_config["max_tokens"])
    samples = class_balance(samples, filter_config["target_safe_ratio"], filter_config["target_unsafe_ratio"])

    print(f"\nAfter filtering: {len(samples)} samples")

    # Split
    print("\nSplitting...")
    train, val, test = stratified_split(
        samples, split_config["train"], split_config["val"], split_config["test"],
    )

    save_jsonl(train, PROCESSED_DIR / "train.jsonl")
    save_jsonl(val, PROCESSED_DIR / "val.jsonl")
    save_jsonl(test, PROCESSED_DIR / "test.jsonl")

    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Category distribution
    print("\nCategory distribution (train):")
    for cat in CATEGORIES:
        count = sum(1 for s in train if s.get("categories", {}).get(cat, False))
        print(f"  {cat}: {count} ({count/len(train)*100:.1f}%)")

    # Source distribution
    print("\nSource distribution (train):")
    source_counts = Counter(s.get("source", "unknown") for s in train)
    for src, count in source_counts.most_common():
        print(f"  {src}: {count} ({count/len(train)*100:.1f}%)")


if __name__ == "__main__":
    main()
