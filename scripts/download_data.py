"""Tải dữ liệu thô CDC Diabetes Health Indicators (UCI dataset #891).

Thư mục ``data/raw/`` bị ``.gitignore`` nên mỗi thành viên phải tự tải dữ liệu.
Script này tải về ``data/raw/diabetes.csv`` rồi kiểm SHA-256 khớp giá trị đã
ghi trong ``docs/data_dictionary.md`` — bảo đảm cả nhóm dùng đúng một file.
File đã tồn tại và đúng hash thì bỏ qua, không tải lại.

Cách chạy (từ thư mục gốc project):
    python scripts/download_data.py            # tải nếu chưa có hoặc sai hash
    python scripts/download_data.py --force    # buộc tải lại

Sau khi tải xong, tái tạo bản sạch bằng:  python -m src.cleaning
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Nguồn + checksum đã chốt trong docs/data_dictionary.md (M02).
DATA_URL = "https://archive.ics.uci.edu/static/public/891/data.csv"
EXPECTED_SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
RAW_PATH = Path("data/raw/diabetes.csv")


def sha256_of(path: Path) -> str:
    """Tính SHA-256 theo khối 1 MB, tránh nạp toàn bộ file vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path) -> None:
    """Tải về file tạm ``*.part`` rồi đổi tên — tránh để lại file dở dang
    nếu mất mạng giữa chừng."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as f:
            while chunk := response.read(1 << 20):
                f.write(chunk)
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Tải raw data UCI #891 và kiểm SHA-256.")
    parser.add_argument("--force", action="store_true",
                        help="tải lại kể cả khi file hiện tại đã đúng hash")
    args = parser.parse_args()

    # File đã có và đúng hash -> không cần làm gì thêm.
    if RAW_PATH.exists() and not args.force:
        digest = sha256_of(RAW_PATH)
        if digest == EXPECTED_SHA256:
            print(f"OK: {RAW_PATH} đã tồn tại và đúng hash, bỏ qua tải.")
            return 0
        print(f"CẢNH BÁO: {RAW_PATH} tồn tại nhưng sai hash ({digest}) — tải lại.")

    print(f"Đang tải {DATA_URL} -> {RAW_PATH} ...")
    try:
        download(DATA_URL, RAW_PATH)
    except Exception as exc:  # mất mạng, UCI chết, DNS lỗi...
        print(f"LỖI tải dữ liệu: {exc}", file=sys.stderr)
        return 1

    digest = sha256_of(RAW_PATH)
    if digest != EXPECTED_SHA256:
        print(f"LỖI: hash thực tế {digest}", file=sys.stderr)
        print(f"     khác hash chốt {EXPECTED_SHA256} — file có thể bị hỏng "
              f"hoặc nguồn UCI đã đổi. Báo lại phụ trách M02.", file=sys.stderr)
        return 1

    print(f"OK: đã tải và kiểm hash đúng ({digest[:16]}...).")
    print("Bước tiếp theo: chạy 'python -m src.cleaning' để tạo bản sạch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
