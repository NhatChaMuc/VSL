from pathlib import Path
from typing import List


def list_dataset_files(data_dir: str | Path) -> List[Path]:
    """Liệt kê các file dữ liệu trong thư mục dataset."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted([p for p in data_path.rglob("*") if p.is_file()])


def load_dataset(data_dir: str | Path):
    """Placeholder cho việc load dữ liệu huấn luyện."""
    files = list_dataset_files(data_dir)
    return files
