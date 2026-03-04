---
license: mit
language:
- en
library_name: transformers
tags:
- safety
- toxicity
- content-moderation
- deberta
- text-classification
- guard-model
datasets:
- lmsys/toxic-chat
- google/civil_comments
- PKU-Alignment/BeaverTails
- allenai/wildguardmix
pipeline_tag: text-classification
model-index:
- name: TinySafe v2
  results:
  - task:
      type: text-classification
      name: Toxicity Detection
    dataset:
      name: ToxicChat
      type: lmsys/toxic-chat
      config: toxicchat0124
      split: test
    metrics:
    - type: f1
      value: 0.782
      name: F1 (Binary)
    - type: recall
      value: 0.798
      name: Unsafe Recall
    - type: precision
      value: 0.767
      name: Unsafe Precision
  - task:
      type: text-classification
      name: Over-Refusal Detection
    dataset:
      name: OR-Bench
      type: bench-llm/or-bench
      config: or-bench-80k
      split: train
    metrics:
    - type: accuracy
      value: 0.962
      name: Safe Accuracy (1 - FPR)
---

# TinySafe v2

![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fjdleo1%2Ftinysafe-2&query=%24.downloads&label=%F0%9F%A4%97%20Monthly%20Downloads&color=blue)
![Parameters](https://img.shields.io/badge/params-141M-orange)
![License](https://img.shields.io/github/license/jdleo/tinysafe-2)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Card-yellow)](https://huggingface.co/jdleo1/tinysafe-2)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)

141M parameter safety classifier built on DeBERTa-v3-small. Binary safe/unsafe classification with 7-category multi-label head (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

Successor to [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) (71M params, 59% TC F1). v2 improves ToxicChat F1 by **+19 points** while cutting OR-Bench false positive rate from 18.9% to 3.8%.

**GitHub:** [jdleo/tinysafe-2](https://github.com/jdleo/tinysafe-2)

## Benchmarks

| Benchmark | TinySafe v2 | TinySafe v1 |
|---|---|---|
| **ToxicChat F1** | 78.2% | 59.2% |
| **OR-Bench FPR** | 3.8% | 18.9% |
| **WildGuardBench F1** | 62.7% | 75.0% |

### ToxicChat Leaderboard

| Model | Params | F1 |
|---|---|---|
| *internal-safety-reasoner (unreleased)* | *unknown* | *81.3%* |
| *gpt-5-thinking (unreleased)* | *unknown* | *81.0%* |
| *gpt-oss-safeguard-20b (unreleased)* | *21B (3.6B\*)* | *79.9%* |
| gpt-oss-safeguard-120b | 117B (5.1B\*) | 79.3% |
| Toxic Prompt RoBERTa | 125M | 78.7% |
| **TinySafe v2** | **141M** | **78.2%** |
| Qwen3Guard-8B | 8B | 73% |
| AprielGuard-8B | 8B | 72% |
| Granite Guardian-8B | 8B | 71% |
| WildGuard | 7B | 70.8% |
| Granite Guardian-3B | 3B | 68% |
| ShieldGemma-2B | 2B | 67% |
| Qwen3Guard-0.6B | 0.6B | 63% |
| [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) | 71M | 59% |

*\* = active params (MoE)*

### OR-Bench (Over-Refusal)

| Model | FPR |
|---|---|
| **TinySafe v2** | **3.8%** |
| WildGuard-7B | ~10% |
| [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) | 18.9% |

Lower is better. On 80K safe prompts, TinySafe v2 incorrectly flags only 3.8%.

## Quickstart

```python
import torch
from transformers import DebertaV2Tokenizer

# Load
tokenizer = DebertaV2Tokenizer.from_pretrained("jdleo1/tinysafe-2")
model = torch.load("model.pt", map_location="cpu")  # or load from checkpoint

# Inference
text = "how do i make a bomb"
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
with torch.no_grad():
    binary_logits, category_logits = model(inputs["input_ids"], inputs["attention_mask"])
    unsafe_score = torch.sigmoid(binary_logits).item()
    print(f"Unsafe: {unsafe_score:.3f}")  # 0.998
```

## Architecture

DeBERTa-v3-small (6 transformer layers, 768 hidden dim) with dual classification heads:

- **Binary head**: single logit (safe/unsafe)
- **Category head**: 7-way multi-label (violence, hate, sexual, self_harm, dangerous_info, harassment, illegal_activity)

Training enhancements:
- **FGM adversarial training** (epsilon=0.3): perturbs embeddings for robustness
- **EMA** (decay=0.999): smoothed weight averaging for stable eval
- **Multi-sample dropout** (5 masks): averaged logits across dropout samples
- **DualHeadLossV2**: focal loss (binary) + asymmetric class-balanced loss (categories)

## Training

Single-phase unified fine-tuning (5 epochs, LR=2e-5) with source-weighted sampling:

| Source | Weight | Samples | Purpose |
|---|---|---|---|
| ToxicChat | 4.0x | ~4K | Anchor benchmark signal |
| WildGuardTrain | 1.0x | ~10K | Adversarial/jailbreak coverage |
| Jigsaw Civil Comments | 0.5x | ~7K | General toxicity diversity |
| BeaverTails | 1.5x | ~2.2K | Behavior-value alignment |
| Hard negatives (Claude) | 1.2x | ~10K | FPR control |

Model selection on val F1 only (no test set leakage).

## Limitations

- **Low-resource categories (violence, hate, sexual) have 0 F1** -- <200 training samples per category is insufficient even with class-balanced loss
- **WildGuardBench generalization is weak** -- encoder-only models struggle with adversarial jailbreak rephrasing
- **Conservative on out-of-distribution inputs** -- high precision but lower recall suggests the model learned narrow patterns rather than general safety reasoning

These are fundamental limitations of encoder-only architectures for safety classification. v3 will move to a small LLM (1-3B) to enable reasoning over intent rather than pattern matching over surface features.

## License

MIT
