#!/usr/bin/env python3
"""
Step 7 (GPU): Evaluate v2 model on all benchmarks.
- Internal test set
- ToxicChat test (primary target)
- WildGuardBench test
- OR-Bench (over-refusal)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer

from src.dataset import SafetyDataset
from src.model import SafetyClassifierV2
from src.utils import CATEGORIES, load_config, normalize_sample, save_jsonl

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def load_model(config, device):
    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=config["num_categories"],
        layers_to_keep=None,  # Full model, no pruning
        num_dropout_samples=config["training"]["multi_sample_dropout_count"],
    )
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).float()
    model.eval()
    return model


def predict_batch(model, dataloader, device):
    all_binary_probs, all_binary_labels = [], []
    all_cat_probs, all_cat_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, category_logits = model(input_ids, attention_mask, multi_sample=False)

            all_binary_probs.extend(torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy())
            all_binary_labels.extend((batch["binary_label"] > 0.5).float().numpy())
            all_cat_probs.extend(torch.sigmoid(category_logits).cpu().numpy())
            all_cat_labels.extend(batch["category_labels"].numpy())

    return (
        np.array(all_binary_probs),
        np.array(all_binary_labels),
        np.array(all_cat_probs),
        np.array(all_cat_labels),
    )


def compute_metrics(binary_probs, binary_labels, cat_probs, cat_labels, threshold=0.5, cat_thresholds=None):
    binary_preds = (binary_probs > threshold).astype(int)

    metrics = {
        "f1_binary": f1_score(binary_labels, binary_preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(binary_labels, binary_preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(binary_labels, binary_preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0),
        "fpr": 1 - recall_score(binary_labels, binary_preds, pos_label=0, zero_division=0),
        "threshold": threshold,
    }

    for i, cat in enumerate(CATEGORIES):
        cat_t = (cat_thresholds or {}).get(cat, 0.5)
        cat_preds = (cat_probs[:, i] > cat_t).astype(int)
        if cat_labels[:, i].sum() > 0:
            metrics[f"{cat}_f1"] = f1_score(cat_labels[:, i], cat_preds, zero_division=0)
            metrics[f"{cat}_precision"] = precision_score(cat_labels[:, i], cat_preds, zero_division=0)
            metrics[f"{cat}_recall"] = recall_score(cat_labels[:, i], cat_preds, zero_division=0)

    return metrics


def eval_benchmark(name, model, tokenizer, config, device, samples, threshold, cat_thresholds, use_cuda):
    print(f"\n{'='*60}")
    print(name)
    print("=" * 60)

    nw = config["training"]["num_workers"] if use_cuda else 0
    tmp_path = Path(f"data/eval/{name.lower().replace(' ', '_')}.jsonl")
    save_jsonl(samples, tmp_path)
    eval_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False,
                        num_workers=nw, pin_memory=use_cuda)
    probs, labels, cat_probs, cat_labels = predict_batch(model, loader, device)
    metrics = compute_metrics(probs, labels, cat_probs, cat_labels, threshold, cat_thresholds)

    print(f"  Samples: {len(samples)}")
    print(f"  F1 (binary):     {metrics['f1_binary']:.4f}")
    print(f"  Unsafe recall:   {metrics['unsafe_recall']:.4f}")
    print(f"  Unsafe precision: {metrics['unsafe_precision']:.4f}")
    print(f"  FPR:             {metrics['fpr']:.4f}")

    for cat in CATEGORIES:
        if f"{cat}_f1" in metrics:
            print(f"  {cat}: F1={metrics[f'{cat}_f1']:.3f} R={metrics[f'{cat}_recall']:.3f}")

    return metrics


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    Path("data/eval").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    use_cuda = device.type == "cuda"
    nw = config["training"]["num_workers"] if use_cuda else 0
    print(f"Using device: {device}")

    threshold = config.get("inference", {}).get("binary_threshold", 0.45)
    cat_thresholds = config.get("inference", {}).get("category_thresholds", {})
    print(f"Binary threshold: {threshold}")
    print(f"Category thresholds: {cat_thresholds}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
    model = load_model(config, device)

    results = {}

    # 1. Internal test
    test_ds = SafetyDataset("data/processed/test.jsonl", tokenizer, config["max_length"])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             num_workers=nw, pin_memory=use_cuda)
    probs, labels, cat_probs, cat_labels = predict_batch(model, test_loader, device)
    results["internal_test"] = compute_metrics(probs, labels, cat_probs, cat_labels, threshold, cat_thresholds)
    print(f"\n{'='*60}")
    print("Internal Test Set")
    print("=" * 60)
    m = results["internal_test"]
    print(f"  F1 (binary): {m['f1_binary']:.4f} | Recall: {m['unsafe_recall']:.4f} | Precision: {m['unsafe_precision']:.4f}")

    # 2. ToxicChat test
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="test")
    tc_samples = []
    for row in ds:
        text = row.get("user_input", "")
        if text.strip():
            label = "unsafe" if row.get("toxicity", 0) == 1 else "safe"
            tc_samples.append(normalize_sample(text, label, source="toxic_chat_test"))
    results["toxic_chat"] = eval_benchmark("ToxicChat Test", model, tokenizer, config, device, tc_samples, threshold, cat_thresholds, use_cuda)

    # 3. WildGuardBench (gated — skip if not authenticated)
    try:
        ds = load_dataset("allenai/wildguardmix", "wildguardtest", split="test")
        wg_samples = []
        for row in ds:
            text = row.get("prompt", "")
            if text.strip():
                label = "unsafe" if row.get("prompt_harm_label", "") == "harmful" else "safe"
                wg_samples.append(normalize_sample(text, label, source="wildguard_test"))
        results["wildguard"] = eval_benchmark("WildGuardBench", model, tokenizer, config, device, wg_samples, threshold, cat_thresholds, use_cuda)
    except Exception as e:
        print(f"\nSkipping WildGuardBench: {e}")

    # 4. OR-Bench
    print(f"\n{'='*60}")
    print("OR-Bench (Over-Refusal)")
    print("=" * 60)
    ds = load_dataset("bench-llm/or-bench", "or-bench-80k", split="train")
    or_samples = []
    for row in ds:
        text = row.get("prompt", "")
        if text.strip():
            or_samples.append(normalize_sample(text, "safe", source="or_bench"))

    tmp_path = Path("data/eval/or_bench.jsonl")
    save_jsonl(or_samples, tmp_path)
    eval_ds = SafetyDataset(tmp_path, tokenizer, config["max_length"])
    loader = DataLoader(eval_ds, batch_size=32, shuffle=False,
                        num_workers=nw, pin_memory=use_cuda)
    probs, labels, _, _ = predict_batch(model, loader, device)
    preds = (probs > threshold).astype(int)
    fpr = preds.sum() / len(preds)
    results["or_bench"] = {"fpr": float(fpr), "total": len(or_samples), "false_positives": int(preds.sum())}
    print(f"  Samples: {len(or_samples)}")
    print(f"  FPR: {fpr*100:.1f}% ({int(preds.sum())}/{len(or_samples)} false positives)")

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print("=" * 60)

    tc = results.get("toxic_chat", {})
    wg = results.get("wildguard", {})
    orb = results.get("or_bench", {})

    header = f"{'Model':<30} {'Params':<10} {'TC F1':<10} {'TC Rec':<10} {'TC Prec':<10} {'WG F1':<10} {'OR FPR':<10}"
    print(header)
    print("-" * len(header))

    or_fpr = f"{orb.get('fpr', 0)*100:.1f}%"
    print(f"{'TinySafe v2.1':<30} {'141M':<10} {tc.get('f1_binary', 0):<10.4f} {tc.get('unsafe_recall', 0):<10.4f} {tc.get('unsafe_precision', 0):<10.4f} {wg.get('f1_binary', 0):<10.4f} {or_fpr:<10}")
    print(f"{'TinySafe v2':<30} {'141M':<10} {'0.782':<10} {'—':<10} {'—':<10} {'0.421':<10} {'3.8%':<10}")
    print(f"{'TinySafe v1':<30} {'71M':<10} {'0.5924':<10} {'0.6243':<10} {'~0.82':<10} {'0.7502':<10} {'18.9%':<10}")
    print(f"{'Intel toxic-prompt-roberta':<30} {'125M':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10} {'—':<10}")
    print(f"{'WildGuard-7B':<30} {'7B':<10} {'~0.92':<10} {'~0.90':<10} {'~0.94':<10} {'~0.90':<10} {'~10%':<10}")

    # Save results
    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    with open(RESULTS_DIR / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nResults saved to {RESULTS_DIR / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
