"""Real-time webcam inference cho nhận diện ngôn ngữ ký hiệu VSL.

Pipeline:
    Camera → MediaPipe Holistic → keypoints (1605,) → buffer 60 frames
    → TransformerClassifier → label VSL → hiển thị OpenCV
"""
from __future__ import annotations

# Đảm bảo project root trong sys.path khi chạy: python realtime/webcam.py
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from realtime.mediapipe_runtime import KEYPOINT_DIM, extract_keypoints, init_mediapipe

# ── Cấu hình hiển thị ────────────────────────────────────────────────────────
_FONT          = cv2.FONT_HERSHEY_SIMPLEX
_COLOR_GREEN   = (0, 220, 80)
_COLOR_YELLOW  = (0, 200, 255)
_COLOR_RED     = (0, 60, 220)
_COLOR_WHITE   = (255, 255, 255)
_COLOR_DARK    = (20, 20, 20)

SEQUENCE_LEN   = 60   # số frame mỗi cửa sổ dự đoán
STRIDE         = 15   # chạy inference mỗi bao nhiêu frame mới (sliding window)
CONFIDENCE_THR = 0.4  # ngưỡng tin cậy tối thiểu để hiển thị kết quả


def _load_label_map(label_map_path: Path | None) -> dict[int, str]:
    """Đọc label_map.json (index → tên ký hiệu VSL).
    Trả về dict rỗng nếu không tìm thấy.
    """
    if label_map_path is None or not Path(label_map_path).exists():
        return {}
    try:
        raw: dict[str, str] = json.loads(Path(label_map_path).read_text(encoding="utf-8"))
        return {int(k): v for k, v in raw.items()}
    except Exception:
        return {}


def _draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    collecting: bool,
    collected: int,
    fps: float,
) -> np.ndarray:
    """Vẽ overlay thông tin lên frame OpenCV."""
    h, w = frame.shape[:2]

    # Nền mờ phía trên
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), _COLOR_DARK, -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    if collecting:
        bar_w = int(w * collected / SEQUENCE_LEN)
        cv2.rectangle(frame, (0, 55), (bar_w, 68), _COLOR_YELLOW, -1)
        cv2.putText(frame, f"Dang thu thap: {collected}/{SEQUENCE_LEN} frames",
                    (10, 45), _FONT, 0.6, _COLOR_YELLOW, 1, cv2.LINE_AA)
    else:
        color = _COLOR_GREEN if confidence >= CONFIDENCE_THR else _COLOR_RED
        cv2.putText(frame, label, (10, 45), _FONT, 1.0, color, 2, cv2.LINE_AA)
        conf_text = f"Confidence: {confidence:.1%}"
        cv2.putText(frame, conf_text, (10, 65), _FONT, 0.5, _COLOR_WHITE, 1, cv2.LINE_AA)

    # FPS góc trên phải
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 22), _FONT, 0.5, _COLOR_WHITE, 1, cv2.LINE_AA)

    # Hướng dẫn góc dưới
    cv2.putText(frame, "Nhan Q de thoat", (10, h - 10), _FONT, 0.45, _COLOR_WHITE, 1, cv2.LINE_AA)
    return frame


def run_webcam(
    checkpoint_path: str | Path = "checkpoints/best.pt",
    label_map_path: str | Path | None = "checkpoints/label_map.json",
    num_classes: int = 0,
    device: str = "cpu",
    camera_index: int = 0,
) -> None:
    """Chạy real-time webcam inference.

    Args:
        checkpoint_path:  Đường dẫn file .pt checkpoint đã train.
        label_map_path:   Đường dẫn label_map.json (index → tên ký hiệu).
                          Nếu None hoặc không tồn tại, hiển thị số index.
        num_classes:      Số class. Nếu = 0, đọc từ checkpoint metadata.
        device:           'cpu' hoặc 'cuda'.
        camera_index:     Camera ID (thường 0 = webcam tích hợp).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}\n"
            "Hãy chạy train.py trước để tạo checkpoint."
        )

    # ── Đọc metadata checkpoint ───────────────────────────────────────────────
    meta_path = checkpoint_path.with_suffix(".json")
    if num_classes == 0 and meta_path.exists():
        try:
            meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
            num_classes = int(meta.get("extra", {}).get("num_classes", 0))
        except Exception:
            pass

    if num_classes == 0:
        raise ValueError(
            "Không xác định được num_classes từ checkpoint.\n"
            "Truyền tham số num_classes= khi gọi run_webcam()."
        )

    label_map = _load_label_map(Path(label_map_path) if label_map_path else None)

    # ── Load model ────────────────────────────────────────────────────────────
    from model.transformer import TransformerClassifier
    model = TransformerClassifier(
        input_dim=KEYPOINT_DIM,   # 1605
        d_model=256,
        num_heads=8,
        num_layers=4,
        ff_dim=512,
        dropout=0.0,
        num_classes=num_classes,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[INFO] Đã load model: {checkpoint_path} ({num_classes} classes)")

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    holistic = init_mediapipe()

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        holistic.close()
        raise RuntimeError(f"Không mở được camera (index={camera_index}).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[INFO] Webcam đang chạy — nhấn Q để thoát.")
    print(f"[INFO] Thu thập {SEQUENCE_LEN} frames rồi dự đoán (stride={STRIDE} frames).")

    # ── Sliding window buffer ─────────────────────────────────────────────────
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LEN)  # mỗi phần tử: (1605,)
    current_label     = "Dang khoi dong..."
    current_conf      = 0.0
    frames_since_pred = 0
    t_prev            = time.perf_counter()

    import mediapipe as mp   # đã lazy import an toàn

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Không đọc được frame từ camera.")
                break

            # FPS
            t_now = time.perf_counter()
            fps = 1.0 / max(1e-6, t_now - t_prev)
            t_prev = t_now

            # MediaPipe xử lý (Tasks API 1.0.0)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results   = holistic.detect(mp_image)

            # Vẽ skeleton thủ công bằng OpenCV
            h_fr, w_fr = frame.shape[:2]

            def _draw_lms(lm_list: list, color: tuple, radius: int = 3) -> None:
                for lm in lm_list:
                    x, y = int(lm.x * w_fr), int(lm.y * h_fr)
                    cv2.circle(frame, (x, y), radius, color, -1)

            if results.pose_landmarks:
                _draw_lms(results.pose_landmarks[0][:25], (245, 117, 66), 3)
            if results.left_hand_landmarks:
                _draw_lms(results.left_hand_landmarks[0], (121, 44, 250), 4)
            if results.right_hand_landmarks:
                _draw_lms(results.right_hand_landmarks[0], (66, 200, 245), 4)

            # Extract keypoints + thêm vào buffer
            kp = extract_keypoints(results)  # (1605,)
            buffer.append(kp)

            collecting = len(buffer) < SEQUENCE_LEN
            frames_since_pred += 1

            # Inference khi buffer đầy và đã đủ stride
            if not collecting and frames_since_pred >= STRIDE:
                frames_since_pred = 0
                seq = np.stack(list(buffer), axis=0)          # (60, 1605)
                tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 60, 1605)

                with torch.no_grad():
                    logits = model(tensor)
                    probs  = torch.softmax(logits, dim=1)
                    pred   = int(torch.argmax(probs, dim=1).item())
                    conf   = float(probs[0, pred].item())

                if conf >= CONFIDENCE_THR:
                    current_conf  = conf
                    current_label = label_map.get(pred, f"[{pred}]")
                else:
                    current_conf  = conf
                    current_label = f"Không chắc ({label_map.get(pred, str(pred))})"

            # Vẽ overlay
            frame = _draw_overlay(
                frame,
                label=current_label,
                confidence=current_conf,
                collecting=collecting,
                collected=len(buffer),
                fps=fps,
            )

            cv2.imshow("VSL Real-time Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        holistic.close()
        cv2.destroyAllWindows()
        print("[INFO] Đã tắt webcam.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-time VSL recognition qua webcam")
    parser.add_argument("--checkpoint",  default="checkpoints/best.pt",        help="Đường dẫn checkpoint .pt")
    parser.add_argument("--label-map",   default="checkpoints/label_map.json",  help="Đường dẫn label_map.json")
    parser.add_argument("--num-classes", type=int, default=0,                   help="Số class (0 = tự đọc từ checkpoint)")
    parser.add_argument("--device",      default="cpu",                         help="cpu hoặc cuda")
    parser.add_argument("--camera",      type=int, default=0,                   help="Camera index (0 = mặc định)")
    args = parser.parse_args()

    run_webcam(
        checkpoint_path=args.checkpoint,
        label_map_path=args.label_map,
        num_classes=args.num_classes,
        device=args.device,
        camera_index=args.camera,
    )
