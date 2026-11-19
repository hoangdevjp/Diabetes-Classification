"""M02 / Phú: làm sạch dữ liệu thô CDC Diabetes Health Indicators.

Phạm vi: chỉ các quy tắc xác định, độc lập với tham số học, nên có thể chạy
trước khi chia train/test 80/20 phân tầng. Các bước học tham số (imputation,
scaling, encoding) thuộc M03 preprocessing trong pipeline của từng training
fold — cố ý KHÔNG cài đặt ở đây.

Các quy tắc (kết quả khảo sát ghi trong docs/data_dictionary.md):
    R1. Bỏ cột ``ID``: index nhân tạo do mirror UCI thêm vào, không phải
        đặc trưng của file gốc trên Kaggle.
    R2. Bỏ các dòng trùng tuyệt đối theo feature+label (24,206 dòng, ~9.5%).
    R3. Bỏ dòng vi phạm miền giá trị hợp lệ đã tài liệu (bản v1: 0 dòng).
    R4. Ép kiểu int64; ép ô không phải số thành NaN để lộ ra lỗi.
    R5. Báo cáo dữ liệu thiếu nhưng không imputation (bản v1: 0 ô thiếu).

Cách chạy:
    python -m src.cleaning
        đọc data/raw/diabetes.csv -> ghi data/processed/diabetes_clean.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/diabetes.csv")
CLEAN_PATH = Path("data/processed/diabetes_clean.csv")

TARGET = "Diabetes_binary"
DROP_COLUMNS = ["ID"]  # R1: cột index do UCI thêm.

# R3: miền giá trị hợp lệ quan sát trên file raw, khớp codebook BRFSS
# (cờ nhị phân, mã ordinal bắt đầu từ 1, số ngày 0-30, BMI nguyên).
BINARY_COLUMNS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
RANGED_COLUMNS = {
    "BMI": (12, 98),          # khoảng hợp lệ lâm sàng; giữ giá trị cực trị
    "GenHlth": (1, 5),        # 1=excellent .. 5=poor
    "MentHlth": (0, 30),      # số ngày trong 30 ngày gần nhất
    "PhysHlth": (0, 30),      # số ngày trong 30 ngày gần nhất
    "Age": (1, 13),           # 13 khoảng tuổi, 1=18-24 .. 13=80+
    "Education": (1, 6),      # bậc học vấn thứ tự
    "Income": (1, 8),         # khoảng thu nhập thứ tự
}

EXPECTED_COLUMNS = ["ID", TARGET] + BINARY_COLUMNS + list(RANGED_COLUMNS)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Áp dụng các quy tắc làm sạch dữ liệu thô. Trả về (df_sạch, báo cáo)."""
    report: dict = {"input_rows": int(len(df)), "steps": []}

    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # R4: ép mọi cột sang số; chuỗi không hợp lệ thành NaN để lộ lỗi.
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # R5: chỉ đếm missing; không imputation ở M02.
    n_missing = int(df.isna().sum().sum())
    report["steps"].append({"rule": "R5_missing_report", "missing_cells": n_missing})
    df = df.dropna(subset=[TARGET])

    # R1: bỏ các cột định danh.
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    report["steps"].append({"rule": "R1_drop_id", "dropped": DROP_COLUMNS})

    # R3: bỏ các dòng nằm ngoài miền giá trị hợp lệ đã tài liệu.
    before = len(df)
    valid = df[TARGET].isin([0, 1])
    for col in BINARY_COLUMNS:
        valid &= df[col].isin([0, 1])
    for col, (lo, hi) in RANGED_COLUMNS.items():
        valid &= df[col].between(lo, hi)
    df = df.loc[valid]
    report["steps"].append(
        {"rule": "R3_range_check", "rows_dropped": int(before - len(df))}
    )

    # R2: bỏ duplicate tuyệt đối (feature + target), giữ bản ghi đầu tiên.
    before = len(df)
    df = df.drop_duplicates()
    report["steps"].append(
        {"rule": "R2_deduplicate", "rows_dropped": int(before - len(df))}
    )

    # R4: ép kiểu int64 cuối cùng (mọi miền giá trị đều là số nguyên)
    # và reset index để đầu ra có RangeIndex liên tục như file CSV.
    df = df.astype("int64").reset_index(drop=True)

    report["output_rows"] = int(len(df))
    report["label_counts"] = {
        str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()
    }
    return df, report


def main() -> int:
    raw = pd.read_csv(RAW_PATH)
    clean_df, report = clean(raw)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_PATH, index=False)
    report["input_path"] = str(RAW_PATH)
    report["output_path"] = str(CLEAN_PATH)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
