import io
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

# Fix UnicodeEncodeError tren Windows (cp932 / cp1252)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import yaml

from data.dataloader import build_dataloader
from data.dataset import VOYADataset
from data.download_dataset import load_voya_dataset
from model.transformer import TransformerClassifier
from trainer.evaluate import evaluate_model
from trainer.trainer import train_model
from utils.checkpoint import save_checkpoint
from utils.seed import set_seed


# ── Helpers đọc config an toàn (tránh Pylance "type unknown") ────────────────

def _cfg_int(cfg: dict[str, Any], *keys: str, default: int = 0) -> int:
    """Đọc int từ config lồng nhau.  Ví dụ: _cfg_int(cfg, 'model', 'd_model')."""
    _node: Any = cfg
    for k in keys:
        if not isinstance(_node, dict) or k not in _node:
            return default
        _node = _node[k]
    return int(_node) if isinstance(_node, (int, float, str)) else default


def _cfg_float(cfg: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Đọc float từ config lồng nhau."""
    _node: Any = cfg
    for k in keys:
        if not isinstance(_node, dict) or k not in _node:
            return default
        _node = _node[k]
    try:
        return float(_node)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    with open("config/config.yaml", "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    set_seed(42)

    # ── 1. Tải dataset ───────────────────────────────────────────────────────
    dataset_name: str = cfg.get("dataset", {}).get("name", "Kateht/VOYA_VSL")
    dataset = load_voya_dataset(dataset_name)

    num_classes: int = dataset.get("num_classes", 0) or _cfg_int(cfg, "model", "num_classes")
    if num_classes == 0:
        raise ValueError(
            "Không xác định được num_classes. "
            "Dataset phải chứa trường 'labels' hoặc config.yaml phải đặt num_classes > 0."
        )
    print(f"[INFO] num_classes      = {num_classes}")

    # ── 2. Tách train / val (80 / 20) ────────────────────────────────────────
    raw_train = dataset["train"]
    max_samples: int = _cfg_int(cfg, "dataset", "max_samples", default=0)
    if max_samples > 0:
        n = min(max_samples, len(raw_train))
        raw_train = raw_train.select(range(n)) if hasattr(raw_train, "select") else raw_train[:n]

    all_samples: list[Any] = list(raw_train)
    random.shuffle(all_samples)

    split_idx         = int(len(all_samples) * 0.8)
    train_samples: list[Any] = all_samples[:split_idx]
    val_samples:   list[Any] = all_samples[split_idx:]
    print(f"[INFO] Tổng mẫu        = {len(all_samples)}  "
          f"(train={len(train_samples)}, val={len(val_samples)})")

    # ── 3. Dataset / DataLoader ──────────────────────────────────────────────
    input_dim:       int = _cfg_int(cfg, "model", "input_dim",       default=1605)
    sequence_length: int = _cfg_int(cfg, "model", "sequence_length", default=60)
    batch_size:      int = _cfg_int(cfg, "training", "batch_size",   default=8)

    train_dataset = VOYADataset(
        train_samples, input_dim=input_dim,
        sequence_length=sequence_length, num_classes=num_classes,
    )
    val_dataset = VOYADataset(
        val_samples, input_dim=input_dim,
        sequence_length=sequence_length, num_classes=num_classes,
    )

    num_workers: int = _cfg_int(cfg, "training", "num_workers", default=0)
    train_loader = build_dataloader(train_dataset, batch_size=batch_size,
                                    shuffle=True,  num_workers=num_workers)
    val_loader   = build_dataloader(val_dataset,   batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

    # ── 4. Thiết bị ──────────────────────────────────────────────────────────
    use_cuda: bool = bool(cfg.get("device", {}).get("use_cuda", False))
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Thiết bị        = {device}")

    # ── 5. Model ─────────────────────────────────────────────────────────────
    model = TransformerClassifier(
        input_dim=input_dim,
        d_model=_cfg_int(cfg,    "model", "d_model",    default=128),
        num_heads=_cfg_int(cfg,   "model", "num_heads",  default=4),
        num_layers=_cfg_int(cfg,  "model", "num_layers", default=2),
        ff_dim=_cfg_int(cfg,      "model", "ff_dim",     default=256),
        dropout=_cfg_float(cfg,   "model", "dropout",    default=0.1),
        num_classes=num_classes,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params    = {total_params:,}")

    # ── 6. Training ──────────────────────────────────────────────────────────
    epochs:       int   = _cfg_int(cfg,   "training", "epochs",        default=10)
    lr:           float = _cfg_float(cfg, "training", "learning_rate", default=3e-4)
    weight_decay: float = _cfg_float(cfg, "training", "weight_decay",  default=1e-4)
    patience:     int   = _cfg_int(cfg,   "training", "patience",      default=5)

    save_dir = Path(cfg.get("checkpoint", {}).get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Epochs          = {epochs}  (early stop patience={patience})")
    print(f"[INFO] LR              = {lr}  |  weight_decay={weight_decay}")
    print()

    model, history = train_model(
        model, train_loader, val_loader,
        epochs=epochs, lr=lr, weight_decay=weight_decay,
        patience=patience, device=device,
        checkpoint_dir=str(save_dir),
    )

    # ── 7. Lưu label_map.json ────────────────────────────────────────────────
    label_map_path = save_dir / "label_map.json"
    label_map: dict[str, str] = {}
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
        ds_raw = load_dataset(dataset_name, split="train")
        feat = ds_raw.features.get("labels")
        if hasattr(feat, "names") and feat.names:
            label_map = {str(i): name for i, name in enumerate(feat.names)}
    except Exception:
        pass
    if not label_map:
        label_map = {str(i): str(i) for i in range(num_classes)}
    label_map_path.write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[INFO] label_map.json  -> {label_map_path}  ({len(label_map)} nhãn)")

    # ── 8. Kết quả cuối ──────────────────────────────────────────────────────
    save_checkpoint(model, save_dir / "last.pt")

    best_val = max((h["val_acc"] for h in history), default=0.0)
    last = history[-1]
    elapsed = time.perf_counter() - t_start

    print(f"\n{'='*52}")
    print(f"  Epoch kết thúc    : {len(history)}/{epochs}")
    print(f"  Loss cuối         : {last['train_loss']:.4f}")
    print(f"  Train acc cuối    : {last['train_acc']:.4f}")
    print(f"  Val acc cuối      : {last['val_acc']:.4f}")
    print(f"  Best val acc      : {best_val:.4f}")
    print(f"  Thời gian         : {elapsed/60:.1f} phút")
    print(f"{'='*52}")

    best_path = save_dir / "best.pt"
    print(f"\n[INFO] Chạy webcam real-time:")
    print(f"  python realtime/webcam.py --checkpoint {best_path} --label-map {label_map_path}")


if __name__ == "__main__":
    main()