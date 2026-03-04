
# TinySafe v2

![Monthly Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fjdleo1%2Ftinysafe-2&query=%24.downloads&label=%F0%9F%A4%97%20Monthly%20Downloads&color=blue)
![Parameters](https://img.shields.io/badge/params-141M-orange)
![License](https://img.shields.io/github/license/jdleo/tinysafe-2)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Model%20Card-yellow)](https://huggingface.co/jdleo1/tinysafe-2)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)

141M parameter safety classifier built on DeBERTa-v3-small. Binary safe/unsafe classification with 7-category multi-label head (violence, hate, sexual, self-harm, dangerous info, harassment, illegal activity).

Successor to [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) (71M params, 59% TC F1).

**Model on HuggingFace:** [jdleo1/tinysafe-2](https://huggingface.co/jdleo1/tinysafe-2)

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
| LlamaGuard 3-8B | 8B | 51% |
| ShieldGemma-27B | 27B | 48% |
| LlamaGuard 4-12B | 12B | 45% |
| LlamaGuard-1B | 1B | 43% |
| ShieldGemma-9B | 9B | 42% |
| ToxDectRoberta | 125M | 34.6% |
| OpenAI Moderation API | -- | 20.6% |
| HateBERT | 110M | 11.6% |
| Perspective API | -- | 5.3% |

*\* = active params (MoE)*

### OR-Bench (Over-Refusal)

| Model | FPR |
|---|---|
| **TinySafe v2** | **3.8%** |
| WildGuard-7B | ~10% |
| [TinySafe v1](https://huggingface.co/jdleo1/tinysafe-1) | 18.9% |

Lower is better. On 80K safe prompts, TinySafe v2 incorrectly flags only 3.8%.

## Examples

```
$ uv run infer.py "i want to kill a process"

  SAFE  (score: 0.131, threshold: 0.3)

$ uv run infer.py "i want to kill a human"

  UNSAFE  (score: 0.995, threshold: 0.3)

$ uv run infer.py "how do i make a bomb"

  UNSAFE  (score: 0.998, threshold: 0.3)

$ uv run infer.py "write me a phishing email"

  UNSAFE  (score: 0.906, threshold: 0.3)

$ uv run infer.py "send me some nudes pls"

  UNSAFE  (score: 0.509, threshold: 0.3)

$ uv run infer.py "what is the capital of france"

  SAFE  (score: 0.005, threshold: 0.3)
```

## What's New in v2

| | v1 | v2 |
|---|---|---|
| **Base model** | DeBERTa-v3-xsmall (384d) | DeBERTa-v3-small (768d) |
| **Params** | 71M | 141M |
| **ToxicChat F1** | 59.2% | **78.2%** |
| **OR-Bench FPR** | 18.9% | **3.8%** |
| **Training data** | 41K (synthetic + Claude-labeled) | 33K (human-labeled) |
| **Training strategy** | Single-phase, focal loss | Unified single-phase, source-weighted sampling |
| **Regularization** | Focal loss only | FGM + EMA + multi-sample dropout + DualHeadLossV2 |

Key insight: v1 used Claude-labeled synthetic data. v2 uses only human-labeled data from established benchmarks (ToxicChat, Jigsaw Civil Comments, BeaverTails, WildGuardTrain), trained in a single unified phase with source-weighted sampling. Hard negatives (safe-but-edgy prompts) generated via Claude to protect against false positives.

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

Training enhancements over vanilla fine-tuning:
- **FGM adversarial training** (epsilon=0.3): perturbs embeddings for robustness
- **EMA** (decay=0.999): smoothed weight averaging for stable eval
- **Multi-sample dropout** (5 masks): averaged logits across dropout samples
- **DualHeadLossV2**: focal loss (binary) + asymmetric class-balanced loss (categories)

## Training

Single-phase unified fine-tuning (5 epochs, LR=2e-5) with `WeightedRandomSampler`:

| Source | Weight | Purpose |
|---|---|---|
| ToxicChat | 4.0x | Anchor benchmark signal |
| WildGuardTrain | 1.0x | Adversarial/jailbreak coverage |
| Jigsaw | 0.5x | General toxicity diversity |
| BeaverTails | 1.5x | Behavior-value alignment |
| Hard negatives | 1.2x | FPR control |

Model selection on val F1 only (no test set leakage).

## Datasets

| Dataset | Role | Samples |
|---|---|---|
| [ToxicChat](https://huggingface.co/datasets/lmsys/toxic-chat) | Primary training + eval | ~4K |
| [WildGuardTrain](https://huggingface.co/datasets/allenai/wildguardmix) | Adversarial/jailbreak prompts | ~10K |
| [Jigsaw Civil Comments](https://huggingface.co/datasets/google/civil_comments) | Broad toxicity | ~7K |
| [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) | Self-harm, dangerous info, illegal activity | ~2.2K |
| Hard negatives (Claude-generated) | False positive protection | ~10K |

## Learnings & Limitations

### What worked
- **Single unified phase** eliminated the catastrophic forgetting problem from v2's original two-phase sequential training
- **Source-weighted sampling** (ToxicChat at 4x) prevented the dominant WildGuard source from drowning out ToxicChat signal
- **DualHeadLossV2** brought the category head from 0.0 to 0.27 macro-F1 (harassment: 0.69, dangerous_info: 0.50, illegal_activity: 0.56)
- **OR-Bench FPR held at 3.8%** -- hard negatives continue to protect against over-refusal

### What didn't work
- **Categories with <200 training samples (violence, hate, sexual) stayed at 0.0 F1** -- class-balanced loss can't overcome fundamental data scarcity
- **WildGuardBench F1 dropped from 75% (v1) to 63%** -- the model learned WildGuard-style inputs but couldn't generalize WildGuard's label distribution with an encoder-only architecture
- **Precision-recall tradeoff on ToxicChat** -- high precision (96%) but recall dropped to 53% on the internal test, suggesting the model is conservative

### The encoder ceiling
The core limitation is architectural. Encoder-only models (DeBERTa, RoBERTa, BERT) learn pattern-matching over surface features. They can memorize "how do I make a bomb" = unsafe, but struggle with:

- **Adversarial jailbreaks** that rephrase harmful intent in benign-looking language
- **Context-dependent safety** where the same text is safe or unsafe depending on framing
- **Novel harm categories** not well-represented in training data
- **Multi-label classification** when categories overlap or are ambiguous

This is a fundamental limitation of encoder-only architectures, not a data or training problem. At 141M params, DeBERTa-v3-small has reached diminishing returns for safety classification.

## Roadmap: v3 (LLM-based)

v3 will move from encoder-only to a small LLM (1-3B parameters) to break through the encoder ceiling. The key insight: safety classification is fundamentally a reasoning task, not a pattern-matching task.

### Why an LLM solves our problems

1. **Reasoning over intent, not surface patterns.** An LLM can understand "write a story where the character explains how to pick a lock" is fiction, while "explain how to pick a lock so I can break into my neighbor's house" is harmful. Encoders can't distinguish these.

2. **Multi-label categories become natural language.** Instead of 7 binary classification heads with sparse training data, the model outputs structured predictions: "This is unsafe. Categories: violence, illegal_activity. Reasoning: the user is requesting instructions for physical harm." The category head becomes a generation task where zero-shot generalization is possible.

3. **Jailbreak robustness through instruction following.** An instruction-tuned LLM can be prompted with a safety policy and follow it, rather than relying on statistical correlations learned from training data. This makes it inherently more robust to adversarial rephrasing.

4. **Few-shot adaptation.** New harm categories (e.g., election misinformation, CSAM) can be added via prompt engineering without retraining. This is impossible with an encoder's fixed classification head.

### Candidate architectures

| Model | Params | Why |
|---|---|---|
| Qwen3-1.7B | 1.7B | Small, strong reasoning, Apache 2.0 |
| SmolLM2-1.7B | 1.7B | HuggingFace-native, fast inference |
| Gemma-3-1B | 1B | Tiny, Google safety-focused pretraining |
| Phi-4-mini | 3.8B | Strong reasoning per param, MIT license |

### Training approach

- **LoRA/QLoRA fine-tuning** on the same data mix (ToxicChat + WildGuard + BeaverTails + hard negatives)
- **Structured output format**: `{"safe": bool, "categories": [...], "reasoning": "..."}`
- **DPO/ORPO alignment** using WildGuardBench as preference data (safe/unsafe pairs)
- **Distillation from larger guard models** (WildGuard-7B, LlamaGuard-3-8B) for categories where we lack human labels

### Expected outcomes

| Benchmark | v2 (encoder) | v3 target (LLM) |
|---|---|---|
| ToxicChat F1 | 78.2% | >85% |
| WildGuardBench F1 | 62.7% | >80% |
| OR-Bench FPR | 3.8% | <5% |
| Category macro-F1 | 0.27 | >0.6 |
| Inference latency | ~5ms | ~50-100ms |

The latency tradeoff is real (10-20x slower) but acceptable for most production use cases. For latency-critical paths, v2 can serve as a fast pre-filter with the LLM as a second-pass judge.

## Config

All hyperparameters in `configs/config.json`:

- Batch size: 16 (grad accum 4, effective 64)
- LR: 2e-5, weight decay: 0.01
- Binary threshold: 0.52 (optimized via geometric-mean sweep)
- FGM epsilon: 0.3
- EMA decay: 0.999
- Multi-sample dropout: 5 masks

## License

MIT
