# Hệ thống nhận diện ngôn ngữ ký hiệu tiếng Việt

> Dự án nghiên cứu khoa học về nhận diện ngôn ngữ ký hiệu tiếng Việt bằng mô hình học sâu, tập trung vào việc huấn luyện một mô hình Transformer và thử nghiệm inference real-time qua webcam.

## Giới thiệu

Dự án này xây dựng một pipeline đầy đủ để:

- tải bộ dữ liệu VOYA_VSL (≈4 364 từ/cụm từ VSL) từ Hugging Face,
- huấn luyện mô hình Transformer phân loại chuỗi keypoints tay (60 frames × 1 605 features),
- lưu checkpoint tốt nhất và bản đồ nhãn (label_map.json),
- chạy **nhận diện real-time** qua webcam bằng MediaPipe Holistic,
- chạy inference trên ảnh hoặc video và lưu kết quả vào video/result.

## Công nghệ chính

- Python
- PyTorch
- Hugging Face datasets
- MediaPipe (keypoint extraction)
- OpenCV
- Pillow
- NumPy
- YAML

## Cấu trúc dự án

```text
VSL_NCKH/
├── checkpoints/          # Checkpoint và label_map.json (tạo ra sau train)
├── config/               # config.yaml và labels.py
├── data/                 # dataset.py, dataloader.py, download_dataset.py
├── model/                # TransformerClassifier
├── realtime/             # MediaPipe runtime + webcam inference
├── tests/                # Unit tests
├── trainer/              # trainer.py, evaluate.py
├── utils/                # checkpoint.py, seed.py, metrics.py
├── train.py              # Script huấn luyện chính
├── inference.py          # Inference trên ảnh/video
├── requirements.txt
├── README.md
└── video/
```

## Cài đặt môi trường

### Local

```bash
python -m venv .venv
.venvScriptsactivate
pip install -r requirements.txt
pip install opencv-python mediapipe
```

### Google Colab

```python
!pip install -r https://raw.githubusercontent.com/<your-repo>/main/requirements.txt
```

Nếu bạn muốn cài thủ công trên Colab, dùng đoạn sau:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip install opencv-python pillow numpy pandas matplotlib tqdm PyYAML scikit-learn datasets
```

## Huấn luyện mô hình

Chạy lệnh sau để train mô hình local:

```bash
python train.py
```

### Điểm nổi bật khi train

- mỗi epoch sẽ in loss và accuracy,
- tự động phát hiện `num_classes` từ dataset (không hardcode),
- tách train/val split 80/20,
- checkpoint từng epoch được lưu trong `checkpoints/epoch_*.pt`,
- checkpoint tốt nhất được lưu tại `checkpoints/best.pt`,
- bản đồ nhãn được lưu tại `checkpoints/label_map.json` (dùng cho webcam).

## Chạy inference trên ảnh hoặc thư mục ảnh

```bash
.venvScriptspython.exe inference.py --input .testsassets --checkpoint .checkpointsbest.pt --output .videoresult --device cpu
```

## Chạy inference trên video

```bash
.venvScriptspython.exe inference.py --input .videoyour_video.mp4 --checkpoint .checkpointsbest.pt --output .videoresult --device cpu
```

## Kết quả đầu ra

Sau khi chạy inference, hệ thống sẽ tạo:

- ảnh kết quả có nhãn được lưu vào video/result,
- file tổng hợp JSON: video/result/results.json.

## Cấu hình

Thông số chính được định nghĩa trong [config/config.yaml](config/config.yaml), bao gồm:

- số lớp, số head, dropout trong mô hình Transformer,
- số epoch và learning rate khi train,
- thư mục lưu checkpoint và output.

## Kiểm thử

Bạn có thể chạy các test nhỏ sau:

```bash
.venvScriptspython.exe tests/test_checkpoint_selection.py
.venvScriptspython.exe tests/test_inference_pipeline.py

# Hoặc chạy cả suite:
python -m unittest discover -s tests -v
```
**Pipeline webcam:**

```
Camera → MediaPipe Holistic → keypoints (1 605,) → buffer 60 frames
       → TransformerClassifier → tên ký hiệu VSL + confidence
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `--checkpoint` | `checkpoints/best.pt` | File checkpoint đã train |
| `--label-map` | `checkpoints/label_map.json` | Bản đồ index → tên ký hiệu |
| `--device` | `cpu` | `cpu` hoặc `cuda` |
| `--camera` | `0` | Index webcam (0 = mặc định) |

Nhấn **Q** để thoát.

## Hướng phát triển tiếp theo
- hoàn thiện khả năng realtime trước tiên là thử bằng webcam
- cải thiện preprocessing dữ liệu,
- dùng early stopping,
- tăng độ chính xác bằng mô hình mạnh hơn (GRU, LSTM, ViT),
- hiển thị phụ đề thời gian thực (subtitle) kèm webcam.
