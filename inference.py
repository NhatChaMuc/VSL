import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from model.transformer import TransformerClassifier
from utils.checkpoint import load_checkpoint


def _load_label_map(label_map_path: Path | None) -> dict[int, str]:
    """Load label_map.json (int index → tên kí hiệu VSL)."""
    if label_map_path is None or not label_map_path.exists():
        return {}
    try:
        raw = json.loads(label_map_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _read_num_classes_from_meta(checkpoint_path: Path) -> int:
    """Đọc num_classes từ best.json cạnh checkpoint."""
    meta = checkpoint_path.with_suffix(".json")
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            nc = data.get("extra", {}).get("num_classes")
            if nc:
                return int(nc)
        except Exception:
            pass
    return 0


def build_feature_sequence_from_image(image_path: str | Path, input_dim: int = 1605, sequence_length: int = 60) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = np.array(image, dtype=np.uint8)

    if image.ndim != 3:
        image = np.stack([image] * 3, axis=-1)

    image = cv2.resize(image, (64, 64))
    image = image.astype(np.float32) / 255.0
    image = image.reshape(-1)

    if image.shape[0] < input_dim:
        image = np.pad(image, (0, input_dim - image.shape[0]))
    else:
        image = image[:input_dim]

    sequence = np.zeros((sequence_length, input_dim), dtype=np.float32)
    for t in range(sequence_length):
        start = (t * image.shape[0]) // sequence_length
        end = ((t + 1) * image.shape[0]) // sequence_length
        chunk = image[start:end]
        if chunk.size == 0:
            chunk = image
        sequence[t, : chunk.size] = chunk[: input_dim]

    return torch.tensor(sequence, dtype=torch.float32)


def predict_from_image(model: torch.nn.Module, image_path: str | Path, device: str = "cpu") -> tuple[int, float]:
    input_dim = getattr(getattr(model, "embedding", None), "in_features", 1605)
    feature_seq = build_feature_sequence_from_image(image_path, input_dim=input_dim, sequence_length=60)
    feature_seq = feature_seq.unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(feature_seq)
        probs = torch.softmax(logits, dim=1)
        pred = int(torch.argmax(probs, dim=1).item())
        confidence = float(probs[0, pred].item())
    return pred, confidence


def save_result_image(input_path: str | Path, output_path: str | Path, label: str, confidence: float) -> None:
    image = Image.open(input_path).convert("RGB")
    image = image.copy()
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.putText(
        image,
        f"Pred: {label} ({confidence:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output_path), image)


def label_for_index(index: int, label_map: dict[int, str] | None = None) -> str:
    """Chuyển index class → tên kí hiệu VSL."""
    if label_map and index in label_map:
        return label_map[index]
    return str(index)


def run_inference(
    input_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    label_map_path: str | Path | None = None,
) -> list[dict[str, object]]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_path)

    # Đọc num_classes từ metadata checkpoint, fallback 10
    num_classes = _read_num_classes_from_meta(checkpoint_path) or 10
    label_map   = _load_label_map(Path(label_map_path) if label_map_path else
                                  checkpoint_path.parent / "label_map.json")

    model = TransformerClassifier(
        input_dim=1605,
        d_model=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.1,
        num_classes=num_classes,
    )
    model = load_checkpoint(model, checkpoint_path)
    model.to(device)

    if input_path.is_dir():
        files = sorted(list(input_path.glob("*")))
        media_files = [p for p in files if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".mp4", ".avi", ".mov"}]
    else:
        media_files = [input_path]

    results: list[dict[str, object]] = []
    for media_file in media_files:
        if media_file.suffix.lower() in {".mp4", ".avi", ".mov"}:
            cap = cv2.VideoCapture(str(media_file))
            if not cap.isOpened():
                print(f"[WARN] Không mở được video: {media_file}")
                continue

            frame_idx = 0
            frame_results: list[tuple[int, str, float]] = []
            while True:
                success, frame = cap.read()
                if not success:
                    break
                frame_path = output_dir / f"{media_file.stem}_frame_{frame_idx:03d}.png"
                cv2.imwrite(str(frame_path), frame)
                pred, confidence = predict_from_image(model, frame_path, device=device)
                label = label_for_index(pred)
                frame_results.append((frame_idx, label, confidence))
                save_result_image(frame_path, frame_path.with_suffix(".pred.png"), label, confidence)
                frame_idx += 1

            cap.release()
            results.append({"source": str(media_file), "frames": frame_results})
        else:
            pred, confidence = predict_from_image(model, media_file, device=device)
            label = label_for_index(pred, label_map)
            output_name = media_file.stem + ".pred.png"
            save_result_image(media_file, output_dir / output_name, label, confidence)
            results.append({"source": str(media_file), "pred": label, "confidence": confidence})

    summary_path = output_dir / "results.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[INFO] Kết quả đã được lưu tại {output_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run inference on images or videos")
    parser.add_argument("--input", required=True, help="Path to an image/video file or a folder containing media")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to model checkpoint")
    parser.add_argument("--output", default="video/result", help="Directory for saved outputs")
    parser.add_argument("--device", default="cpu", help="Device to run inference on")
    args = parser.parse_args()

    run_inference(args.input, args.checkpoint, args.output, device=args.device)


if __name__ == "__main__":
    main()
