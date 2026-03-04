#!/usr/bin/env python3
"""
Step 6 (GPU): Binary + per-category threshold optimization.
Sweeps on val set only. Selects threshold maximizing geometric mean of per-source F1s
to prevent overfitting to any single domain.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import SafetyClassifierV2
from src.dataset import SafetyDataset
from src.utils import CATEGORIES, load_config

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")


def get_predictions(model, dataloader, device):
    model.eval()
    all_binary_probs, all_binary_labels = [], []
    all_cat_probs, all_cat_labels = [], []
    all_sources = []

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


def geometric_mean(values):
    """Compute geometric mean of positive values. Returns 0 if any value is 0."""
    values = [v for v in values if v > 0]
    if not values:
        return 0.0
    return np.exp(np.mean(np.log(values)))


def sweep_binary_threshold_multisource(probs, labels, sources, name="val"):
    """Sweep thresholds, select by geometric mean of per-source F1s."""
    thresholds = np.arange(0.25, 0.61, 0.01)
    results = []
    best_geomean = 0
    best_threshold = 0.5

    # Get unique source prefixes for grouping
    source_groups = defaultdict(list)
    for i, src in enumerate(sources):
        # Group by source prefix (e.g., "jigsaw_ub" and "jigsaw_tc" stay separate)
        source_groups[src].append(i)

    print(f"\nSources in val set: {', '.join(f'{k}({len(v)})' for k, v in sorted(source_groups.items()))}")
    print(f"\n{'Thresh':<8} {'Overall':<10} {'GeoMean':<10} ", end="")
    for src in sorted(source_groups.keys()):
        print(f"{src[:12]:<14}", end="")
    print(f" {'FPR':<8}")
    print("-" * (28 + 14 * len(source_groups) + 8))

    for t in thresholds:
        preds = (probs > t).astype(int)
        overall_f1 = f1_score(labels, preds, average="binary", zero_division=0)
        fpr = 1 - recall_score(labels, preds, pos_label=0, zero_division=0)

        # Per-source F1
        source_f1s = {}
        for src, indices in sorted(source_groups.items()):
            idx = np.array(indices)
            src_labels = labels[idx]
            src_preds = preds[idx]
            # Only compute F1 if source has both classes
            if src_labels.sum() > 0 and (1 - src_labels).sum() > 0:
                source_f1s[src] = f1_score(src_labels, src_preds, average="binary", zero_division=0)
            elif src_labels.sum() > 0:
                # All unsafe — use recall as proxy
                source_f1s[src] = recall_score(src_labels, src_preds, pos_label=1, zero_division=0)
            else:
                # All safe — use safe recall (1 - FPR) as proxy
                source_f1s[src] = recall_score(src_labels, src_preds, pos_label=0, zero_division=0)

        gm = geometric_mean(list(source_f1s.values())) if source_f1s else 0.0

        marker = ""
        if gm > best_geomean:
            best_geomean = gm
            best_threshold = t
            marker = " *"

        print(f"{t:<8.2f} {overall_f1:<10.4f} {gm:<10.4f} ", end="")
        for src in sorted(source_groups.keys()):
            print(f"{source_f1s.get(src, 0):<14.4f}", end="")
        print(f" {fpr:<8.4f}{marker}")

        results.append({
            "threshold": round(float(t), 2),
            "f1_binary": float(overall_f1),
            "geometric_mean_f1": float(gm),
            "per_source_f1": {k: float(v) for k, v in source_f1s.items()},
            "fpr": float(fpr),
        })

    print(f"\nBest {name} geometric-mean F1: {best_geomean:.4f} at threshold {best_threshold:.2f}")

    # Also report overall F1 at best threshold for reference
    preds = (probs > best_threshold).astype(int)
    overall_at_best = f1_score(labels, preds, average="binary", zero_division=0)
    print(f"Overall F1 at best threshold: {overall_at_best:.4f}")

    return best_threshold, best_geomean, results


def sweep_category_thresholds(cat_probs, cat_labels):
    thresholds = np.arange(0.20, 0.71, 0.05)
    best_thresholds = {}

    print(f"\nPer-category threshold sweep:")
    print(f"{'Category':<18} {'Best Thresh':<15} {'F1':<10} {'Recall':<10} {'Prec':<10}")
    print("-" * 63)

    for i, cat in enumerate(CATEGORIES):
        if cat_labels[:, i].sum() == 0:
            best_thresholds[cat] = 0.5
            print(f"{cat:<18} {'N/A':<15} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            continue

        best_f1 = 0
        best_t = 0.5
        best_rec = 0
        best_prec = 0

        for t in thresholds:
            preds = (cat_probs[:, i] > t).astype(int)
            f1 = f1_score(cat_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
                best_rec = recall_score(cat_labels[:, i], preds, zero_division=0)
                best_prec = precision_score(cat_labels[:, i], preds, zero_division=0)

        best_thresholds[cat] = round(float(best_t), 2)
        print(f"{cat:<18} {best_t:<15.2f} {best_f1:<10.4f} {best_rec:<10.4f} {best_prec:<10.4f}")

    return best_thresholds


def main():
    config = load_config()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])

    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=config["num_categories"],
        layers_to_keep=None,
        num_dropout_samples=config["training"]["multi_sample_dropout_count"],
    )
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).float()
    model.eval()

    use_cuda = device.type == "cuda"
    nw = config["training"]["num_workers"] if use_cuda else 0

    # Val set sweep only (no test set — prevents leakage)
    val_ds = SafetyDataset("data/processed/val.jsonl", tokenizer, config["max_length"])
    val_loader = DataLoader(val_ds, batch_size=32,
                            shuffle=False, num_workers=nw, pin_memory=use_cuda)

    # Extract source labels from val set for per-source F1
    val_sources = [s.get("source", "unknown") for s in val_ds.samples]

    print("=" * 60)
    print("Val Set — Multi-Source Binary Threshold Sweep (0.25 - 0.60)")
    print("=" * 60)
    binary_probs, binary_labels, cat_probs, cat_labels = get_predictions(model, val_loader, device)
    best_val_t, best_geomean, val_results = sweep_binary_threshold_multisource(
        binary_probs, binary_labels, val_sources, "val"
    )

    # Per-category threshold sweep (on val set)
    print("\n" + "=" * 60)
    print("Per-Category Threshold Sweep (Val Set)")
    print("=" * 60)
    best_cat_thresholds = sweep_category_thresholds(cat_probs, cat_labels)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Best threshold (geo-mean): {best_val_t:.2f} (geo-mean F1={best_geomean:.4f})")
    print(f"  Category thresholds:       {best_cat_thresholds}")

    # Save results
    output = {
        "best_binary_threshold": round(float(best_val_t), 2),
        "best_geometric_mean_f1": float(best_geomean),
        "category_thresholds": best_cat_thresholds,
        "val_sweep": val_results,
    }
    with open(RESULTS_DIR / "threshold_sweep.json", "w") as f:
        json.dump(output, f, indent=2)

    # Update config with best thresholds
    config["inference"]["binary_threshold"] = round(float(best_val_t), 2)
    config["inference"]["category_thresholds"] = best_cat_thresholds
    with open("configs/config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nUpdated configs/config.json with optimized thresholds")
    print(f"Results saved to {RESULTS_DIR / 'threshold_sweep.json'}")


if __name__ == "__main__":
    main()
