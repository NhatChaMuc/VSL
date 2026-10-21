import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str | torch.device = "cpu",
) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    accuracy = correct / max(1, total)
    print(f"[INFO] Evaluation accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy}
