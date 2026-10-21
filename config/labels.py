# Danh sách nhãn theo bảng chữ cái VSL tiếng Việt.
# Đây là mapping thực, không phải hash ngẫu nhiên.

VSL_ALPHABET: list[str] = [
    "a",
    "ă",
    "â",
    "b",
    "c",
    "d",
    "đ",
    "e",
    "ê",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "ô",
    "ơ",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "ư",
    "v",
    "x",
    "y",
]

# Alias thực tế để đảm bảo nhập liệu có thể là chữ cái | chữ cái + vùng miền | chữ viết hoa | cách tên khác.
REAL_LABEL_ALIASES: dict[str, str] = {
    "a": "a",
    "ă": "ă",
    "â": "â",
    "b": "b",
    "c": "c",
    "d": "d",
    "đ": "đ",
    "e": "e",
    "ê": "ê",
    "g": "g",
    "h": "h",
    "i": "i",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ô": "ô",
    "ơ": "ơ",
    "p": "p",
    "q": "q",
    "r": "r",
    "s": "s",
    "t": "t",
    "u": "u",
    "ư": "ư",
    "v": "v",
    "x": "x",
    "y": "y",
    "chữ a": "a",
    "chữ ă": "ă",
    "chữ â": "â",
    "chữ b": "b",
    "chữ c": "c",
    "chữ d": "d",
    "chữ đ": "đ",
    "chữ e": "e",
    "chữ ê": "ê",
    "chữ g": "g",
    "chữ h": "h",
    "chữ i": "i",
    "chữ k": "k",
    "chữ l": "l",
    "chữ m": "m",
    "chữ n": "n",
    "chữ o": "o",
    "chữ ô": "ô",
    "chữ ơ": "ơ",
    "chữ p": "p",
    "chữ q": "q",
    "chữ r": "r",
    "chữ s": "s",
    "chữ t": "t",
    "chữ u": "u",
    "chữ ư": "ư",
    "chữ v": "v",
    "chữ x": "x",
    "chữ y": "y",
}

LABELS: list[str] = VSL_ALPHABET[:]
LABEL_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}


def normalize_vsl_label(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        idx = int(value)
        if 0 <= idx < len(LABELS):
            return LABELS[idx]
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    # Loại bỏ dấu gạch nối/underscore và khoảng trắng thừa
    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    # Khớp chính xác với chữ cái hoặc alias quy ước
    canonical = REAL_LABEL_ALIASES.get(text)
    if canonical is not None:
        return canonical

    # Nếu label đang là form "a (bắc)" hoặc "ê (nam)" thì lấy chữ cái đầu tiên
    if text and text[0] in "abcdefghijklmnopqrstuvwxyzăâêôơưđ":
        base = text.split()[0]
        if base in LABEL_TO_INDEX:
            return base

    return None
