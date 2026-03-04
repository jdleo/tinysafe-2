#!/usr/bin/env python3
"""
Step 5 (GPU): Single-phase unified training with source-weighted sampling.
All sources mixed every epoch — eliminates catastrophic forgetting from sequential phases.

Uses DualHeadLossV2 (focal + asymmetric + class-balanced) to revive the category head.
Model selection on val F1 only (no test set leakage).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import DebertaV2Tokenizer
from sklearn.metrics import f1_score, precision_score, recall_score

from src.model import SafetyClassifierV2, EMAModel, FGM
from src.dataset import SafetyDataset
from src.losses import DualHeadLossV2
from src.utils import CATEGORIES, load_config, load_jsonl

CHECKPOINT_DIR = Path("checkpoints")

# Per-source sampling weights: controls how often each source appears in batches
SOURCE_WEIGHTS = {
    "toxicchat": 4.0,        # Oversample — anchor benchmark, only ~7% of data
    "wildguard_train": 1.0,  # Largest source, natural weight
    "jigsaw_ub": 0.5,        # Downsample — different domain (forum comments)
    "jigsaw_tc": 0.5,        # Downsample — same domain as jigsaw_ub
    "beavertails": 1.5,      # Moderate boost — behavior-value alignment
    "hard_neg": 1.2,         # Slight boost — FPR control
}


def get_source_weight(source: str) -> float:
    """Get sampling weight for a source, matching by prefix."""
    for prefix, weight in SOURCE_WEIGHTS.items():
        if source.startswith(prefix):
            return weight
    return 1.0  # default


def build_weighted_sampler(dataset: SafetyDataset) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler based on per-source weights."""
    weights = []
    for sample in dataset.samples:
        source = sample.get("source", "unknown")
        weights.append(get_source_weight(source))

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


def evaluate(model, dataloader, device, threshold=0.5):
    """Evaluate model, return metrics dict."""
    model.eval()
    all_probs, all_labels = [], []
    all_cat_probs, all_cat_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            binary_logits, cat_logits = model(input_ids, attention_mask, multi_sample=False)

            all_probs.extend(torch.sigmoid(binary_logits.squeeze(-1)).cpu().numpy())
            all_labels.extend(batch["binary_label"].numpy())
            all_cat_probs.extend(torch.sigmoid(cat_logits).cpu().numpy())
            all_cat_labels.extend(batch["category_labels"].numpy())

    probs = np.array(all_probs)
    labels = (np.array(all_labels) > 0.5).astype(int)
    preds = (probs > threshold).astype(int)

    metrics = {
        "f1_binary": f1_score(labels, preds, average="binary", zero_division=0),
        "unsafe_recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "unsafe_precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "safe_recall": recall_score(labels, preds, pos_label=0, zero_division=0),
    }

    cat_probs = np.array(all_cat_probs)
    cat_labels = np.array(all_cat_labels)
    cat_f1s = []
    for i, cat in enumerate(CATEGORIES):
        if cat_labels[:, i].sum() > 0:
            cat_preds = (cat_probs[:, i] > 0.5).astype(int)
            f1 = f1_score(cat_labels[:, i], cat_preds, zero_division=0)
            metrics[f"{cat}_f1"] = f1
            cat_f1s.append(f1)

    if cat_f1s:
        metrics["category_macro_f1"] = np.mean(cat_f1s)

    return metrics


def train_one_epoch(model, loader, optimizer, scheduler, device, grad_accum,
                    fgm, ema, loss_fn):
    """Single epoch with DualHeadLossV2, FGM, EMA, gradient clipping."""
    model.train()
    total_loss = 0
    total_binary_loss = 0
    total_cat_loss = 0
    step_count = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        binary_labels = batch["binary_label"].to(device).float()
        category_labels = batch["category_labels"].to(device).float()

        binary_logits, cat_logits = model(input_ids, attention_mask)

        loss_dict = loss_fn(binary_logits, cat_logits, binary_labels, category_labels)
        loss = loss_dict["loss"] / grad_accum
        loss.backward()

        # FGM adversarial training
        if fgm is not None:
            fgm.attack()
            adv_binary, adv_cat = model(input_ids, attention_mask)
            adv_loss_dict = loss_fn(adv_binary, adv_cat, binary_labels, category_labels)
            (adv_loss_dict["loss"] / grad_accum).backward()
            fgm.restore()

        total_loss += loss_dict["loss"].item()
        total_binary_loss += loss_dict["binary_loss"].item()
        total_cat_loss += loss_dict["category_loss"].item()
        step_count += 1

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if ema is not None:
                ema.update(model)

        if (step + 1) % 100 == 0:
            avg = total_loss / step_count
            avg_b = total_binary_loss / step_count
            avg_c = total_cat_loss / step_count
            lr = scheduler.get_last_lr()[0]
            print(f"    Step {step+1}/{len(loader)} | Loss: {avg:.4f} (bin={avg_b:.4f}, cat={avg_c:.4f}) | LR: {lr:.2e}")

    return total_loss / max(step_count, 1)


def main():
    config = load_config()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    use_cuda = device.type == "cuda"
    nw = 8 if use_cuda else 0

    tokenizer = DebertaV2Tokenizer.from_pretrained(config["base_model"])
    max_len = config["max_length"]
    num_cats = config["num_categories"]

    # Load training data
    train_ds = SafetyDataset(Path("data/processed/train.jsonl"), tokenizer, max_len)
    print(f"Total training samples: {len(train_ds)}")

    # Source distribution
    from collections import Counter
    source_counts = Counter(s.get("source", "unknown") for s in train_ds.samples)
    print("Source distribution:")
    for src, count in source_counts.most_common():
        w = get_source_weight(src)
        print(f"  {src}: {count} (weight={w:.1f})")

    # Val loader (val-only model selection — no test leakage)
    val_ds = SafetyDataset(Path("data/processed/val.jsonl"), tokenizer, max_len)
    val_loader = DataLoader(val_ds, batch_size=32,
                            shuffle=False, num_workers=nw, pin_memory=use_cuda)

    # Build model — full DeBERTa-v3-small, no pruning
    model = SafetyClassifierV2(
        base_model_name=config["base_model"],
        num_categories=num_cats,
        layers_to_keep=None,
        num_dropout_samples=config["training"]["multi_sample_dropout_count"],
        dropout_rate=0.1,
    ).to(device).float()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # DualHeadLossV2 with class-balanced category loss
    category_counts = train_ds.get_category_counts()
    print(f"Category counts: {dict(zip(CATEGORIES, category_counts))}")

    loss_fn = DualHeadLossV2(
        gamma=config["training"]["focal_loss_gamma"],
        label_smoothing=config["training"]["label_smoothing"],
        category_weight=config["training"]["category_loss_weight"],
        asl_gamma_pos=config["training"]["asl_gamma_pos"],
        asl_gamma_neg=config["training"]["asl_gamma_neg"],
        asl_clip=config["training"]["asl_clip"],
        samples_per_class=category_counts,
    ).to(device)

    # FGM adversarial training
    fgm = FGM(model, epsilon=config["training"]["fgm_epsilon"])

    # Weighted sampler
    sampler = build_weighted_sampler(train_ds)

    # Training config — single unified phase
    num_epochs = 5
    lr = 2e-5
    batch_size = 16 if use_cuda else 32
    grad_accum = 4  # effective batch size = 16 * 4 = 64

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=nw, pin_memory=use_cuda,
    )

    total_steps = len(train_loader) * num_epochs // grad_accum
    warmup_steps = int(total_steps * 0.06)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        total_steps=total_steps, pct_start=max(warmup_steps / total_steps, 0.01),
    )

    ema_tracker = EMAModel(model, decay=config["training"]["ema_decay"])
    best_val_f1 = 0

    print(f"\n{'='*60}")
    print(f"UNIFIED TRAINING — {num_epochs} epochs, LR={lr}, BS={batch_size}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            grad_accum, fgm, ema_tracker, loss_fn,
        )
        elapsed = time.time() - t0
        print(f"  Train loss: {train_loss:.4f} ({elapsed:.1f}s)")

        # Eval with EMA weights — free training memory first
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ema_tracker.apply_shadow(model)
        val_metrics = evaluate(model, val_loader, device)

        print(f"  Val F1: {val_metrics['f1_binary']:.4f} | "
              f"Recall: {val_metrics['unsafe_recall']:.4f} | "
              f"Prec: {val_metrics['unsafe_precision']:.4f} | "
              f"Safe-R: {val_metrics['safe_recall']:.4f}")
        if "category_macro_f1" in val_metrics:
            print(f"  Category macro-F1: {val_metrics['category_macro_f1']:.4f}")
            for cat in CATEGORIES:
                if f"{cat}_f1" in val_metrics:
                    print(f"    {cat}: {val_metrics[f'{cat}_f1']:.3f}")

        # Model selection on val F1 only
        current = val_metrics["f1_binary"]
        if current > best_val_f1:
            best_val_f1 = current
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {"training": "unified"},
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  * New best val F1: {best_val_f1:.4f}")

        ema_tracker.restore(model)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
