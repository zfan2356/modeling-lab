# Lab 1 Results

Date: 2026-06-21

## Environment

- Container: `modeling-lab-dev`
- GPU: NVIDIA GeForce RTX 5070 Ti
- PyTorch: `2.12.1+cu130`
- Device: `cuda`

## Experiment 1: Weight Decay Sweep

Fixed setup:

```bash
uv run python ml/deep-learning/algorithmic-grokking/lab1/train.py \
  --p 31 \
  --train-fraction 0.3 \
  --steps 3000 \
  --eval-every 300 \
  --batch-size 256 \
  --d-model 64 \
  --heads 4 \
  --layers 2 \
  --seed 1
```

Only `--weight-decay` was changed.

| weight_decay | final loss | train_acc | test_acc | observation |
| --- | ---: | ---: | ---: | --- |
| 0.0 | 0.000874 | 1.0000 | 0.2912 | Memorizes train set quickly, no grokking in 3000 steps. |
| 0.1 | 0.001535 | 1.0000 | 0.2927 | Same behavior as no decay at this horizon. |
| 1.0 | 1.156826 | 0.8160 | 0.1842 | Decay is too strong here; it disrupts even training accuracy. |

Key trace for `weight_decay=0.0`:

```text
step 1:    train_acc=0.0486 test_acc=0.0267
step 300:  train_acc=1.0000 test_acc=0.2912
step 3000: train_acc=1.0000 test_acc=0.2912
```

Interpretation: with only 30% of all pairs seen during training, the model learns a lookup-style solution very quickly. Test accuracy plateaus far above random chance but far below algorithmic generalization.

## Experiment 2: Training Coverage Sweep

Fixed setup:

```bash
uv run python ml/deep-learning/algorithmic-grokking/lab1/train.py \
  --p 31 \
  --steps 3000 \
  --eval-every 300 \
  --batch-size 256 \
  --d-model 64 \
  --heads 4 \
  --layers 2 \
  --seed 1
```

| train_fraction | weight_decay | final loss | train_acc | test_acc | observation |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0.3 | 0.0 | 0.000874 | 1.0000 | 0.2912 | Sparse data: memorization dominates. |
| 0.8 | 0.0 | 0.001119 | 1.0000 | 0.7565 | Dense data improves held-out accuracy, but still not exact rule learning. |
| 0.8 | 0.1 | 0.001910 | 1.0000 | 0.7565 | Weight decay has no visible effect in this short run. |

Interpretation: increasing coverage from 30% to 80% improves test accuracy substantially. This is not enough to claim grokking: the model may still be interpolating from much denser coverage rather than learning a clean modular addition algorithm.

## Fourier Embedding Notes

The top embedding frequencies were stable in the short runs. For example:

```text
wd=0.0, train_fraction=0.3, step 3000:
k=1: 0.080, k=11: 0.078, k=2: 0.076, k=15: 0.073, k=12: 0.070

wd=0.0, train_fraction=0.8, step 3000:
k=1: 0.082, k=11: 0.080, k=2: 0.078, k=15: 0.073, k=12: 0.070
```

This suggests that the current horizon is mostly observing memorization/interpolation, not a strong reorganization of number-token embeddings into a small set of dominant Fourier modes.

## Experiment 3: Longer Weight Decay Sweep

I added structured CSV logging to `train.py` with:

```bash
--metrics-csv <path>
```

The CSV records:

```text
step,batch_loss,train_loss,test_loss,train_acc,test_acc,
train_margin,test_margin,param_norm,embedding_norm,top_embedding_frequencies
```

Fixed setup:

```bash
uv run python ml/deep-learning/algorithmic-grokking/lab1/train.py \
  --p 31 \
  --train-fraction 0.3 \
  --steps 20000 \
  --eval-every 1000 \
  --batch-size 256 \
  --d-model 64 \
  --heads 4 \
  --layers 2 \
  --seed 1
```

| weight_decay | step | train_acc | test_acc | train_margin | test_margin | param_norm | embedding_norm |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 20000 | 1.0000 | 0.2897 | 8.3147 | -3.5620 | 23.6231 | 2.4381 |
| 0.3 | 20000 | 1.0000 | 0.2912 | 11.6636 | -3.0551 | 18.0015 | 1.3344 |
| 0.5 | 20000 | 1.0000 | 0.2912 | 11.0778 | -2.7858 | 17.2332 | 0.4082 |

Interpretation: 20000 steps is still too short for the sparse 30% split. All three runs have memorized the training set, but the test margin is still negative and test accuracy remains near 29%.

The useful signal is that higher weight decay lowers parameter and embedding norms. At `weight_decay=0.5`, the top Fourier frequency starts to concentrate more strongly:

```text
step 20000:
k=11: 0.123, k=8: 0.089, k=1: 0.081, k=6: 0.079, k=12: 0.075
```

This motivated a longer run at `weight_decay=0.5`.

## Experiment 4: Very Long Run at Weight Decay 0.5

Command:

```bash
uv run python ml/deep-learning/algorithmic-grokking/lab1/train.py \
  --p 31 \
  --train-fraction 0.3 \
  --steps 100000 \
  --eval-every 5000 \
  --batch-size 256 \
  --d-model 64 \
  --heads 4 \
  --layers 2 \
  --weight-decay 0.5 \
  --seed 1 \
  --metrics-csv ml/deep-learning/algorithmic-grokking/lab1/runs/very_long_wd_0.5_p31.csv
```

Key trace:

| step | train_acc | test_acc | train_margin | test_margin | param_norm | embedding_norm | note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 45000 | 1.0000 | 0.3195 | 6.6109 | -1.4627 | 20.6401 | 0.1211 | Still mostly memorization. |
| 55000 | 1.0000 | 0.4874 | 7.8698 | 0.7444 | 19.2093 | 0.0786 | Test margin turns positive. |
| 70000 | 1.0000 | 0.8648 | 7.5381 | 3.6253 | 18.4078 | 0.0844 | Clear delayed generalization. |
| 80000 | 1.0000 | 0.9941 | 8.2346 | 6.1309 | 17.7680 | 0.0639 | Near-perfect generalization. |
| 90000 | 1.0000 | 1.0000 | 8.4382 | 7.0655 | 16.9761 | 0.0582 | Perfect test accuracy. |
| 95000 | 1.0000 | 1.0000 | 8.0508 | 7.0298 | 16.4044 | 0.0706 | Perfect test accuracy persists. |
| 100000 | 0.1840 | 0.0386 | -1.0384 | -2.0730 | 17.3582 | 0.1116 | Training collapses under continued constant-LR updates. |

Interpretation: this run shows the grokking pattern clearly:

```text
train accuracy reaches 1.0 early
test accuracy stays low for tens of thousands of steps
test margin becomes positive around 55k
test accuracy rapidly rises between 55k and 90k
```

The final 100k point is a useful caution: with constant learning rate and strong weight decay, the model can leave the good solution after reaching it. A real training script should save best checkpoints or decay the learning rate after the transition.

Fourier structure also becomes more concentrated during the transition:

```text
step 20000:
k=11: 0.123, k=8: 0.089, k=1: 0.081, k=6: 0.079, k=12: 0.075

step 90000:
k=8: 0.256, k=1: 0.241, k=6: 0.202, k=11: 0.132, k=9: 0.031

step 95000:
k=8: 0.284, k=1: 0.233, k=6: 0.194, k=11: 0.152, k=15: 0.026
```

This is consistent with the model moving from a memorization-heavy solution toward a representation that uses a smaller number of cyclic Fourier modes.

## Raw Logs

- `runs/wd_0.0_p31.log`
- `runs/wd_0.1_p31.log`
- `runs/wd_1.0_p31.log`
- `runs/train_frac_0.8_wd_0.0_p31.log`
- `runs/train_frac_0.8_wd_0.1_p31.log`
- `runs/long_wd_0.2_p31.log`
- `runs/long_wd_0.2_p31.csv`
- `runs/long_wd_0.3_p31.log`
- `runs/long_wd_0.3_p31.csv`
- `runs/long_wd_0.5_p31.log`
- `runs/long_wd_0.5_p31.csv`
- `runs/very_long_wd_0.5_p31.log`
- `runs/very_long_wd_0.5_p31.csv`

## Next Experiments

1. Add best-checkpoint saving so the 90k solution is preserved before collapse.
2. Add a learning-rate schedule and rerun `weight_decay=0.5`.
3. Plot `test_acc`, `test_margin`, `param_norm`, and Fourier energy over steps.
4. Repeat with seeds `2`, `3`, and `4` to separate robust behavior from seed luck.
5. Replace the PyTorch built-in Transformer with a hand-written block to inspect attention maps and residual-stream activations.
