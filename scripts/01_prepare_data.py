#!/usr/bin/env python3
"""
Step 1: Download and prepare ToxicChat + Jigsaw + WildGuard datasets.
- ToxicChat train split (human-annotated, our target benchmark)
- Jigsaw Unintended Bias (1.8M comments, continuous toxicity scores)
- Jigsaw Toxic Comment Classification (multi-label categories)
- WildGuardTrain (adversarial/jailbreak prompts from allenai/wildguardmix)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from src.utils import CATEGORIES, JIGSAW_CATEGORY_MAP, load_config, normalize_sample, save_jsonl

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"


def prepare_toxicchat(config: dict) -> list[dict]:
    """Load ToxicChat train split with human labels."""
    print("\n=== ToxicChat ===")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")

    samples = []
    for row in ds:
        text = row.get("user_input", "")
        if not text.strip():
            continue
        label = "unsafe" if row.get("toxicity", 0) == 1 else "safe"
        samples.append(normalize_sample(text, label, source="toxicchat"))

    safe = sum(1 for s in samples if s["label"] == "safe")
    print(f"  Loaded: {len(samples)} (safe={safe}, unsafe={len(samples)-safe})")
    return samples


def prepare_jigsaw_unintended_bias(config: dict) -> list[dict]:
    """Load Jigsaw Unintended Bias with continuous toxicity scores as soft labels."""
    print("\n=== Jigsaw Unintended Bias ===")

    jigsaw_config = config["jigsaw"]
    threshold = jigsaw_config["toxicity_threshold"]
    max_samples = jigsaw_config["max_samples"]

    # google/civil_comments is the HuggingFace version of Jigsaw Unintended Bias
    # Fields: text, toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit
    try:
        ds = load_dataset("google/civil_comments", split="train")
        print(f"  Loaded {len(ds)} rows from google/civil_comments")
    except Exception as e:
        print(f"  ERROR: Cannot load civil_comments dataset: {e}")
        return []

    samples = []
    for row in tqdm(ds, desc="  Processing Jigsaw UB"):
        text = row.get("text", "")
        if not text or not text.strip():
            continue

        toxicity = float(row.get("toxicity", 0.0))
        # Keep everything — soft labels handle ambiguity
        label = "unsafe" if toxicity >= threshold else "safe"

        # Map Jigsaw sub-categories to ours
        cats = {c: False for c in CATEGORIES}
        if toxicity >= threshold:
            if row.get("threat", 0) >= 0.5:
                cats["violence"] = True
            if row.get("identity_attack", 0) >= 0.5:
                cats["hate"] = True
            if row.get("sexual_explicit", 0) >= 0.5:
                cats["sexual"] = True
            if row.get("insult", 0) >= 0.5:
                cats["harassment"] = True
            if row.get("obscene", 0) >= 0.5:
                cats["harassment"] = True

        sample = normalize_sample(text, label, cats, source="jigsaw_ub")
        # Store continuous score as soft label
        if config["jigsaw"]["use_soft_labels"]:
            sample["soft_label"] = toxicity
        samples.append(sample)

    # Subsample to max_samples: keep ALL unsafe, downsample safe
    # Prioritize safe samples near the boundary (toxicity 0.1-0.49) — most informative
    import random
    random.seed(42)
    safe = [s for s in samples if s["label"] == "safe"]
    unsafe = [s for s in samples if s["label"] == "unsafe"]
    print(f"  Before subsample: {len(safe)} safe, {len(unsafe)} unsafe")

    # Cap unsafe if needed
    if len(unsafe) > max_samples // 2:
        # Keep highest toxicity scores first
        unsafe.sort(key=lambda s: s.get("soft_label", 1.0), reverse=True)
        unsafe = unsafe[:max_samples // 2]

    n_safe = max_samples - len(unsafe)
    if len(safe) > n_safe:
        # Prioritize boundary-adjacent safe samples (most useful for learning)
        safe_boundary = [s for s in safe if s.get("soft_label", 0) >= 0.1]
        safe_clear = [s for s in safe if s.get("soft_label", 0) < 0.1]
        random.shuffle(safe_boundary)
        random.shuffle(safe_clear)
        # Take boundary samples first, then fill with clear safe
        safe = (safe_boundary + safe_clear)[:n_safe]

    samples = safe + unsafe
    random.shuffle(samples)

    safe = sum(1 for s in samples if s["label"] == "safe")
    print(f"  Loaded: {len(samples)} (safe={safe}, unsafe={len(samples)-safe})")
    return samples


def prepare_jigsaw_toxic_comments(config: dict) -> list[dict]:
    """Load Jigsaw Toxic Comment Classification for multi-label categories."""
    print("\n=== Jigsaw Toxic Comment Classification ===")

    try:
        ds = load_dataset("google/jigsaw_toxicity_pred", split="train")
        print(f"  Loaded {len(ds)} rows from google/jigsaw_toxicity_pred")
    except Exception as e:
        print(f"  WARNING: Cannot load Jigsaw Toxic Comment dataset: {e}. Skipping.")
        print("  (This is optional — civil_comments is the main Jigsaw source)")
        return []

    samples = []
    for row in tqdm(ds, desc="  Processing Jigsaw TC"):
        text = row.get("comment_text", "")
        if not text or not text.strip():
            continue

        # Check if any toxic category is flagged
        is_toxic = any(row.get(cat, 0) == 1 for cat in JIGSAW_CATEGORY_MAP)
        label = "unsafe" if is_toxic else "safe"

        # Only keep toxic samples (we have plenty of safe from Jigsaw UB)
        if not is_toxic:
            continue

        cats = {c: False for c in CATEGORIES}
        for jigsaw_cat, our_cat in JIGSAW_CATEGORY_MAP.items():
            if our_cat and row.get(jigsaw_cat, 0) == 1:
                cats[our_cat] = True

        samples.append(normalize_sample(text, label, cats, source="jigsaw_tc"))

    print(f"  Loaded: {len(samples)} (all unsafe, multi-labeled)")

    # Category distribution
    for cat in CATEGORIES:
        count = sum(1 for s in samples if s["categories"].get(cat))
        if count > 0:
            print(f"    {cat}: {count}")

    return samples


WILDGUARD_CATEGORY_MAP = {
    "Violence": "violence",
    "Hate": "hate",
    "Sexual": "sexual",
    "Self-Harm": "self_harm",
    "Guns/Illegal Weapons": "dangerous_info",
    "Regulated Substances": "dangerous_info",
    "Suicide & Self Harm": "self_harm",
    "Criminal Planning": "illegal_activity",
    "Privacy": "harassment",
    "Harassment": "harassment",
    "Threat": "violence",
    "Profanity": "harassment",
    "Sexual (minors)": "sexual",
    "Non-Violent Crimes": "illegal_activity",
    "Cybercrime & Unauthorized Intrusion": "illegal_activity",
    "Chemical & Biological Weapons": "dangerous_info",
    "Indiscriminate Weapons": "dangerous_info",
    "Intellectual Property": "illegal_activity",
    "Misinformation & Disinformation": "dangerous_info",
    "Defamation": "harassment",
    "Elections": None,  # not a safety category
    "Tailored Financial Advice": None,
    "Health Consultations": None,
    "Legal Opinions": None,
    "Government Decisions": None,
}


def prepare_wildguard_train(config: dict) -> list[dict]:
    """Load WildGuardTrain split — adversarial/jailbreak prompts for generalization."""
    import random
    random.seed(42)

    print("\n=== WildGuard Train ===")
    max_samples = config.get("wildguard", {}).get("max_samples", 15000)

    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtrain", split="train")
        print(f"  Loaded {len(ds)} rows from allenai/wildguardmix (wildguardtrain)")
    except Exception as e:
        print(f"  ERROR: Cannot load WildGuard dataset: {e}")
        return []

    samples = []
    for row in tqdm(ds, desc="  Processing WildGuard"):
        text = row.get("prompt", "") or row.get("instruction", "") or ""
        if not text.strip():
            continue

        harm_label = row.get("prompt_harm_label", "")
        label = "unsafe" if harm_label == "harmful" else "safe"

        # Map WildGuard harm categories to ours
        cats = {c: False for c in CATEGORIES}
        harm_category = row.get("harm_category", "") or ""
        if label == "unsafe" and harm_category:
            # WildGuard uses comma-separated categories
            for wg_cat in harm_category.split(","):
                wg_cat = wg_cat.strip()
                our_cat = WILDGUARD_CATEGORY_MAP.get(wg_cat)
                if our_cat:
                    cats[our_cat] = True

        samples.append(normalize_sample(text, label, cats, source="wildguard_train"))

    safe = [s for s in samples if s["label"] == "safe"]
    unsafe = [s for s in samples if s["label"] == "unsafe"]
    print(f"  Before subsample: {len(safe)} safe, {len(unsafe)} unsafe")

    # Keep all unsafe, cap safe to hit max_samples
    if len(unsafe) + len(safe) > max_samples:
        n_safe = max_samples - len(unsafe)
        if n_safe > 0 and len(safe) > n_safe:
            safe = random.sample(safe, n_safe)
        elif n_safe <= 0:
            # Too many unsafe — cap unsafe too
            unsafe = random.sample(unsafe, max_samples)
            safe = []

    samples = safe + unsafe
    random.shuffle(samples)

    safe_count = sum(1 for s in samples if s["label"] == "safe")
    print(f"  After subsample: {len(samples)} (safe={safe_count}, unsafe={len(samples)-safe_count})")

    # Category distribution
    for cat in CATEGORIES:
        count = sum(1 for s in samples if s["categories"].get(cat))
        if count > 0:
            print(f"    {cat}: {count}")

    return samples


def main():
    config = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tc_samples = prepare_toxicchat(config)
    jigsaw_ub_samples = prepare_jigsaw_unintended_bias(config)
    jigsaw_tc_samples = prepare_jigsaw_toxic_comments(config)
    wildguard_samples = prepare_wildguard_train(config)

    # Save each source separately
    save_jsonl(tc_samples, RAW_DIR / "toxicchat_train.jsonl")
    save_jsonl(jigsaw_ub_samples, RAW_DIR / "jigsaw_ub.jsonl")
    save_jsonl(jigsaw_tc_samples, RAW_DIR / "jigsaw_tc.jsonl")
    save_jsonl(wildguard_samples, RAW_DIR / "wildguard_train.jsonl")

    # Combined
    all_samples = tc_samples + jigsaw_ub_samples + jigsaw_tc_samples + wildguard_samples
    save_jsonl(all_samples, RAW_DIR / "all_base_data.jsonl")

    safe = sum(1 for s in all_samples if s["label"] == "safe")
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(all_samples)} samples")
    print(f"  Safe: {safe} ({safe/len(all_samples)*100:.1f}%)")
    print(f"  Unsafe: {len(all_samples)-safe} ({(len(all_samples)-safe)/len(all_samples)*100:.1f}%)")
    print(f"  Sources: ToxicChat={len(tc_samples)}, Jigsaw UB={len(jigsaw_ub_samples)}, "
          f"Jigsaw TC={len(jigsaw_tc_samples)}, WildGuard={len(wildguard_samples)}")
    print(f"Saved to {RAW_DIR}")


if __name__ == "__main__":
    main()
