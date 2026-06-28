# Lab 1: LoRA Fine-Tuning with Qwen3.5-2B-Base

## Why This Lab

This lab is the entry point for parameter-efficient fine-tuning, using LoRA on a small local model that fits the available workstation.

Target model:

- `Qwen/Qwen3.5-2B-Base`
- 2B language model parameters
- Apache-2.0 license
- Suitable for LoRA-style PEFT experiments

Local model metadata is mirrored in:

```text
model-info/qwen3.5-2b-base/
```

The full checkpoint weights are intentionally not committed to this repository. See:

```text
model-info/qwen3.5-2b-base/WEIGHTS.md
```

Local hardware target:

- NVIDIA GeForce RTX 5070 Ti
- 16GB VRAM
- bf16 support

This is a practical deep-learning lab, not part of the traditional ML track. It belongs here because LoRA is a modern fine-tuning method for large neural networks, while the traditional ML track should still start from decision trees, linear models, validation, and ensembles.

## Learning Goals

1. Understand why full fine-tuning is expensive.
2. Understand LoRA as a low-rank update to frozen linear layers.
3. Compare LoRA and QLoRA memory tradeoffs.
4. Prepare a tiny supervised fine-tuning dataset.
5. Train, save, reload, and evaluate a LoRA adapter.
6. Learn what can and cannot be concluded from a tiny local fine-tuning run.

## Conceptual Path

The lab should build up in this order:

1. Start with ordinary fine-tuning: all model parameters receive gradients.
2. Freeze the base model and identify the linear layers.
3. Replace a full weight update with a low-rank update:

```text
W' = W + BA
```

where `W` is frozen, and only `A` and `B` are trainable.

4. Train only the adapter parameters.
5. Save only the adapter.
6. Load the frozen base model plus adapter for inference.

## Suggested First Experiment

Use a small sanity dataset first, not a serious instruction-tuning dataset.

The first goal is to prove the training path works:

```text
base model -> tokenizer -> small dataset -> LoRA adapter -> save -> reload -> inference
```

Recommended starting constraints:

- Model: `Qwen/Qwen3.5-2B-Base`
- Precision: bf16
- Sequence length: 1024 or 2048
- Micro batch size: 1 or 2
- Gradient accumulation: 8 to 16
- LoRA rank: 8 or 16
- Target modules: start with `all-linear`

## Future Files

This directory is intentionally only a chapter scaffold for now. A later implementation can add:

```text
data/
configs/
train_lora.py
infer_lora.py
RESULTS.md
```

The first runnable version should stay small enough to complete locally and should record memory usage, throughput, and before/after model behavior.
