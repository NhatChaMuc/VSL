"""MediaPipe Holistic keypoint extraction — MediaPipe >= 1.0.0 (Tasks API).

Dataset VOYA_VSL dùng 1605 features/frame:
    - 25 pose landmarks (upper body)  × 3 (x, y, z) = 75
    - 468 face landmarks              × 3 (x, y, z) = 1404
    - 21 left hand landmarks          × 3 (x, y, z) = 63
    - 21 right hand landmarks         × 3 (x, y, z) = 63
    Tổng: (25 + 468 + 21 + 21) × 3 = 535 × 3 = 1605
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# ── Hằng số — KHÔNG import mediapipe ở đây để module luôn importable ────────
_N_POSE = 25    # chỉ lấy upper body (pose_landmarks[:25])
_N_FACE = 468   # new API trả về 478, ta chỉ lấy 468 đầu để khớp training data
_N_HAND = 21

KEYPOINT_DIM: int = (_N_POSE + _N_FACE + _N_HAND * 2) * 3   # = 1605

# URL model holistic chính thức của Google
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
)
_DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "holistic_landmarker.task"


def _download_model(model_path: Path) -> None:
    """Tải model holistic_landmarker.task nếu chưa có."""
    import urllib.request
    print(f"[INFO] Model chưa có tại {model_path}")
    print(f"[INFO] Đang tải từ {_MODEL_URL} ...")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, str(model_path))
    print("[INFO] Tải xong model MediaPipe.")


def init_mediapipe(
    model_path: str | Path | None = None,
    min_pose_confidence: float = 0.5,
    min_hand_confidence: float = 0.5,
):
    """Khởi tạo HolisticLandmarker (MediaPipe Tasks API >= 1.0.0).

    Args:
        model_path: Đường dẫn file .task. Nếu None, tải tự động về ~/.cache/mediapipe/.
        min_pose_confidence: Ngưỡng tin cậy pose detection.
        min_hand_confidence: Ngưỡng tin cậy hand detection.

    Returns:
        HolisticLandmarker object (gọi .detect() hoặc dùng như context manager).
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except ImportError as exc:
        raise ImportError(
            "MediaPipe chưa được cài đặt. Chạy: pip install mediapipe"
        ) from exc

    path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
    if not path.exists():
        _download_model(path)

    options = vision.HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=min_pose_confidence,
        min_hand_landmarks_confidence=min_hand_confidence,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def extract_keypoints(results) -> np.ndarray:
    """Trích xuất và flatten 1605 keypoints từ kết quả HolisticLandmarker.

    Args:
        results: Kết quả từ holistic.detect(mp_image) — HolisticLandmarkerResult.

    Returns:
        np.ndarray shape (1605,) dtype float32.
        Điền 0.0 cho bộ phận không phát hiện được.
    """
    def _extract(lm_list: list, n: int) -> np.ndarray:
        if lm_list:
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list[:n]], dtype=np.float32).flatten()
        else:
            arr = np.zeros(n * 3, dtype=np.float32)
        # Pad nếu thiếu (ví dụ pose chỉ trả về < 25 landmarks)
        if len(arr) < n * 3:
            arr = np.pad(arr, (0, n * 3 - len(arr)))
        return arr

    # Tasks API: pose_landmarks là list[list[NormalizedLandmark]] (per-person)
    # Lấy người đầu tiên trong frame
    pose_lms = results.pose_landmarks[0]   if results.pose_landmarks   else []
    face_lms = results.face_landmarks[0]   if results.face_landmarks   else []
    lh_lms   = results.left_hand_landmarks[0]  if results.left_hand_landmarks  else []
    rh_lms   = results.right_hand_landmarks[0] if results.right_hand_landmarks else []

    pose = _extract(pose_lms, _N_POSE)
    face = _extract(face_lms, _N_FACE)   # new API: 478 landmarks, ta lấy 468 đầu
    lh   = _extract(lh_lms,   _N_HAND)
    rh   = _extract(rh_lms,   _N_HAND)

    keypoints = np.concatenate([pose, face, lh, rh])
    assert keypoints.shape == (KEYPOINT_DIM,), f"Unexpected shape: {keypoints.shape}"
    return keypoints
