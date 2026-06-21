from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Batch:
    x: torch.Tensor
    y: torch.Tensor


class TinyTransformer(nn.Module):
    def __init__(
        self,
        p: int,
        d_model: int,
        n_heads: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.p = p
        self.op_token = p
        self.token = nn.Embedding(p + 1, d_model)
        self.pos = nn.Parameter(torch.randn(3, d_model) / math.sqrt(d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, p, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token(x) + self.pos
        h = self.blocks(h)
        h = self.norm(h[:, -1])
        return self.head(h)


def make_dataset(p: int, train_fraction: float, seed: int, device: torch.device) -> tuple[Batch, Batch]:
    pairs = [(a, b) for a in range(p) for b in range(p)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    split = int(len(pairs) * train_fraction)

    def pack(items: list[tuple[int, int]]) -> Batch:
        x = torch.tensor([[a, b, p] for a, b in items], dtype=torch.long, device=device)
        y = torch.tensor([(a + b) % p for a, b in items], dtype=torch.long, device=device)
        return Batch(x=x, y=y)

    return pack(pairs[:split]), pack(pairs[split:])


def sample_batch(data: Batch, batch_size: int) -> Batch:
    idx = torch.randint(0, data.x.shape[0], (batch_size,), device=data.x.device)
    return Batch(x=data.x[idx], y=data.y[idx])


@torch.no_grad()
def evaluate(model: nn.Module, data: Batch, batch_size: int = 4096) -> tuple[float, float, float]:
    model.eval()
    correct = 0
    loss_sum = 0.0
    margin_sum = 0.0
    total = data.x.shape[0]
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        y = data.y[start:stop]
        logits = model(data.x[start:stop])
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        true_logits = logits.gather(dim=-1, index=y[:, None]).squeeze(-1)
        other_logits = logits.masked_fill(
            F.one_hot(y, num_classes=logits.shape[-1]).bool(),
            float("-inf"),
        )
        margin_sum += (true_logits - other_logits.max(dim=-1).values).sum().item()
    model.train()
    return loss_sum / total, correct / total, margin_sum / total


@torch.no_grad()
def parameter_norm(model: nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for param in model.parameters():
        total += param.detach().pow(2).sum()
    return total.sqrt().item()


@torch.no_grad()
def embedding_norm(model: TinyTransformer) -> float:
    return model.token.weight[: model.p].detach().norm(dim=-1).mean().item()


@torch.no_grad()
def fourier_embedding_report(model: TinyTransformer, top_k: int) -> str:
    # Only inspect number tokens 0..p-1, excluding the operation token.
    emb = model.token.weight[: model.p].detach()
    emb = emb - emb.mean(dim=0, keepdim=True)
    spectrum = torch.fft.rfft(emb, dim=0)
    energy = (spectrum.abs() ** 2).mean(dim=1)
    energy[0] = 0
    values, indices = torch.topk(energy, k=min(top_k, energy.numel()))
    total = energy.sum().clamp_min(1e-12)
    parts = []
    for idx, value in zip(indices.tolist(), values.tolist(), strict=True):
        parts.append(f"k={idx}: {value / total.item():.3f}")
    return ", ".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train-fraction", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--metrics-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train, test = make_dataset(args.p, args.train_fraction, args.seed, device)
    model = TinyTransformer(
        p=args.p,
        d_model=args.d_model,
        n_heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"device={device} p={args.p} train={train.x.shape[0]} test={test.x.shape[0]}")
    print("step,loss,train_acc,test_acc,top_embedding_frequencies")
    csv_file = None
    writer = None
    if args.metrics_csv is not None:
        args.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = args.metrics_csv.open("w", newline="", encoding="utf-8")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "step",
                "batch_loss",
                "train_loss",
                "test_loss",
                "train_acc",
                "test_acc",
                "train_margin",
                "test_margin",
                "param_norm",
                "embedding_norm",
                "top_embedding_frequencies",
            ],
        )
        writer.writeheader()

    try:
        for step in range(1, args.steps + 1):
            batch = sample_batch(train, args.batch_size)
            logits = model(batch.x)
            loss = F.cross_entropy(logits, batch.y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                train_loss, train_acc, train_margin = evaluate(model, train)
                test_loss, test_acc, test_margin = evaluate(model, test)
                p_norm = parameter_norm(model)
                e_norm = embedding_norm(model)
                freqs = fourier_embedding_report(model, top_k=5)
                print(f"{step},{loss.item():.6f},{train_acc:.4f},{test_acc:.4f},{freqs}")
                if writer is not None:
                    writer.writerow(
                        {
                            "step": step,
                            "batch_loss": f"{loss.item():.8f}",
                            "train_loss": f"{train_loss:.8f}",
                            "test_loss": f"{test_loss:.8f}",
                            "train_acc": f"{train_acc:.8f}",
                            "test_acc": f"{test_acc:.8f}",
                            "train_margin": f"{train_margin:.8f}",
                            "test_margin": f"{test_margin:.8f}",
                            "param_norm": f"{p_norm:.8f}",
                            "embedding_norm": f"{e_norm:.8f}",
                            "top_embedding_frequencies": freqs,
                        }
                    )
    finally:
        if csv_file is not None:
            csv_file.close()


if __name__ == "__main__":
    main()
