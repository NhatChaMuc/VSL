"""Training loop với early stopping, cosine LR scheduler và tqdm progress bar."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer.loss import build_loss
from utils.checkpoint import save_best_checkpoint

try:
    from tqdm import tqdm  # type: ignore[import-untyped]
    _TQDM = True
except ImportError:
    _TQDM = False


def _cosine_lr(optimizer: torch.optim.Optimizer,
               epoch: int, epochs: int,
               base_lr: float, min_lr: float,
               warmup_epochs: int) -> float:
    """Cosine annealing với linear warmup. Trả về lr hiện tại."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
    else:
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader | None = None,
    *,
    epochs:       int   = 10,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    min_lr:       float = 1e-5,
    warmup_epochs: int  = 2,
    patience:     int   = 5,
    device: str | torch.device = "cpu",
    checkpoint_dir: str | None = None,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Train TransformerClassifier.

    Returns:
        (model, history) — history là list dict mỗi epoch:
            {"epoch", "train_loss", "train_acc", "val_acc", "lr"}
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = build_loss()

    model.to(device)
    history: list[dict[str, Any]] = []
    best_val_acc  = float("-inf")
    no_improve    = 0

    for epoch in range(epochs):
        # ── LR schedule ────────────────────────────────────────────────────
        current_lr = _cosine_lr(optimizer, epoch, epochs, lr, min_lr, warmup_epochs)

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss    = 0.0
        running_correct = 0
        total           = 0

        iterator = (
            tqdm(train_loader, desc=f"Epoch {epoch+1:>3}/{epochs}", leave=False, ncols=80)
            if _TQDM else train_loader
        )

        for inputs, targets in iterator:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = inputs.size(0)
            running_loss    += loss.item() * bs
            running_correct += (outputs.argmax(dim=1) == targets).sum().item()
            total           += bs

        train_loss = running_loss    / max(1, total)
        train_acc  = running_correct / max(1, total)

        # ── Val ────────────────────────────────────────────────────────────
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            correct = 0
            n_val   = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs  = inputs.to(device)
                    targets = targets.to(device)
                    preds   = model(inputs).argmax(dim=1)
                    correct += (preds == targets).sum().item()
                    n_val   += targets.size(0)
            val_acc = correct / max(1, n_val)

        row: dict[str, Any] = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_acc":    round(val_acc,    5),
            "lr":         round(current_lr, 7),
        }
        history.append(row)

        print(
            f"Epoch {epoch+1:>3}/{epochs}  "
            f"loss={train_loss:.4f}  "
            f"acc={train_acc:.4f}  "
            f"val_acc={val_acc:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if checkpoint_dir is not None:
            saved = save_best_checkpoint(
                model,
                f"{checkpoint_dir}/best.pt",
                score=val_acc if val_loader is not None else train_acc,
                epoch=epoch + 1,
                extra={
                    "train_loss": train_loss,
                    "train_acc":  train_acc,
                    "val_acc":    val_acc,
                    "num_classes": getattr(
                        getattr(model, "classifier", None), "out_features", None
                    ),
                },
            )
            if saved:
                print(f"  -> Best checkpoint saved (val_acc={val_acc:.4f})")

        # ── Early stopping ─────────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
        else:
            no_improve += 1

        if patience > 0 and no_improve >= patience:
            print(f"\n[INFO] Early stopping sau {epoch+1} epochs "
                  f"(no improvement for {patience} epochs).")
            break

    return model, history
