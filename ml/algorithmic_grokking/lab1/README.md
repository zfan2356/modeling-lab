# Lab 1: Grokking Modular Addition with a Tiny Transformer

## Why This Case

This lab studies a small but surprisingly rich phenomenon: a neural network first memorizes a sparse training set, then much later discovers an algorithmic rule that generalizes. The task is modular addition:

```text
input:  a, b
target: (a + b) mod p
```

The model is intentionally tiny, but the case touches real themes from modern machine learning:

- representation learning in finite groups
- implicit regularization and weight decay
- delayed generalization, also called grokking
- Fourier structure in learned embeddings
- train/test dynamics that are not explained by loss alone

This is a good first ML lab because it is compact enough to fully control, but not merely introductory.

## Research Questions

1. When does the model memorize, and when does it generalize?
2. How do `train_fraction`, `weight_decay`, and model size change the grokking delay?
3. Do learned token embeddings develop Fourier-like structure over the cyclic group `Z_p`?
4. Does a Transformer solve this task differently from an MLP with comparable parameter count?
5. What diagnostics detect algorithmic generalization before test accuracy jumps?

## Run

From inside the development container:

```bash
cd /workspace/modeling-lab
uv run python ml/algorithmic_grokking/lab1/train.py
```

For a faster smoke run:

```bash
uv run python ml/algorithmic_grokking/lab1/train.py --steps 500 --eval-every 100
```

For a more serious run:

```bash
uv run python ml/algorithmic_grokking/lab1/train.py \
  --p 97 \
  --train-fraction 0.3 \
  --steps 20000 \
  --weight-decay 1.0 \
  --d-model 128 \
  --layers 2
```

To save structured metrics for plotting:

```bash
uv run python ml/algorithmic_grokking/lab1/train.py \
  --p 31 \
  --train-fraction 0.3 \
  --steps 20000 \
  --eval-every 1000 \
  --weight-decay 0.5 \
  --metrics-csv ml/algorithmic_grokking/lab1/runs/example.csv
```

## Suggested Experiments

Start with these sweeps:

```bash
uv run python ml/algorithmic_grokking/lab1/train.py --weight-decay 0.0
uv run python ml/algorithmic_grokking/lab1/train.py --weight-decay 0.1
uv run python ml/algorithmic_grokking/lab1/train.py --weight-decay 1.0
```

Then vary data scarcity:

```bash
uv run python ml/algorithmic_grokking/lab1/train.py --train-fraction 0.2
uv run python ml/algorithmic_grokking/lab1/train.py --train-fraction 0.5
uv run python ml/algorithmic_grokking/lab1/train.py --train-fraction 0.8
```

## What To Look For

The important observation is not just final accuracy. Track the phase transition:

- training accuracy reaches high values first
- test accuracy may stay low for many steps
- with the right regularization, test accuracy suddenly rises
- embedding Fourier energy often concentrates in a few frequencies

The lab script prints the top Fourier frequencies of the learned number-token embeddings. Treat those as a starting diagnostic, not a proof.

## Follow-up Labs

Good next labs in this topic:

- `lab2`: compare Transformer, MLP, and bilinear models on the same modular task
- `lab3`: inspect attention heads and residual stream directions
- `lab4`: connect the experiment to Fourier analysis on finite groups
