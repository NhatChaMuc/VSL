import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    torch.save(model.state_dict(), path)


def load_checkpoint(model: torch.nn.Module, path: str | Path) -> torch.nn.Module:
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def save_best_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    score: float,
    epoch: int,
    extra: dict[str, Any] | None = None,
) -> bool:
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    should_update = True

    if path.exists() and metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            previous_score = metadata.get("score", float("-inf"))
            should_update = score > previous_score
        except Exception:
            should_update = True

    if should_update:
        torch.save(model.state_dict(), path)
        payload = {
            "epoch": epoch,
            "score": float(score),
            "extra": extra or {},
        }
        metadata_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return True

    return False
