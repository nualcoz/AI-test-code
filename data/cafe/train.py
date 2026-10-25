"""
train.py
Script huấn luyện và kiểm thử mô hình dự báo doanh thu chuỗi quán Cafe.

Cách chạy:
    python train.py

Nhiệm vụ:
    1. Đọc dữ liệu từ data/ (train.csv, stores.csv, weather.csv, holidays.csv)
    2. Kiểm định chéo qua 4 quý năm 2025 (Q1, Q2, Q3, Q4)
    3. Huấn luyện mô hình toàn diện trên toàn bộ dữ liệu 2024-2025
    4. Lưu trọng số mô hình vào models/cafe_revenue_model.pkl và models/baseline_model.pkl
"""

import sys
import numpy as np
import pandas as pd

# Thiết lập encoding UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    TRAIN_PATH,
    STORES_PATH,
    WEATHER_PATH,
    HOLIDAYS_PATH,
    CAFE_MODEL_PATH,
    BASELINE_MODEL_PATH,
    MODEL_PARAMS,
    EVAL_QUARTERS,
)
from src.cleaning import clean_train, clean_stores
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model import CafeRevenueModel


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính Root Mean Squared Logarithmic Error."""
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def main():
    print("=" * 80)
    print("HUAN LUYEN & KIEM THU MO HINH DU BAO DOANH THU CAFE (RMSLE)")
    print("=" * 80)

    # 1. Đọc dữ liệu thô
    print("\n[1/4] Dang doc du lieu tu data/ ...")
    train_raw = pd.read_csv(TRAIN_PATH)
    stores_raw = pd.read_csv(STORES_PATH)
    weather_raw = pd.read_csv(WEATHER_PATH)
    holidays_raw = pd.read_csv(HOLIDAYS_PATH)
    print(f"  + train.csv: {train_raw.shape[0]:,} dong")
    print(f"  + stores.csv: {stores_raw.shape[0]:,} cua hang")
    print(f"  + weather.csv: {weather_raw.shape[0]:,} quan sat")
    print(f"  + holidays.csv: {holidays_raw.shape[0]:,} su kien/ngay le")

    # 2. Đánh giá kiểm định chéo theo quý (Out-of-Time Quarter Validation)
    print("\n[2/4] Danh gia Kiem dinh cheo theo 4 quy nam 2025...")
    train_c_all, stats = clean_train(train_raw, clean_stores(stores_raw))
    print(f"  + Thong ke lam sach: {stats}")

    results = []
    for q_name, (val_start, val_end) in EVAL_QUARTERS.items():
        tr_split = train_raw[train_raw["date"] < val_start].copy()
        va_raw = train_raw[
            (train_raw["date"] >= val_start) & (train_raw["date"] <= val_end)
        ].copy()
        va_clean = train_c_all[
            (train_c_all["date"] >= val_start) & (train_c_all["date"] <= val_end)
        ].copy()

        # Baseline gốc
        base_raw = BaselineWeekdayModel().fit(tr_split)
        p_base_raw = base_raw.predict(va_raw)
        p_base_clean = base_raw.predict(va_clean)
        score_base_raw = rmsle(va_raw["revenue"].values, p_base_raw)
        score_base_clean = rmsle(va_clean["revenue"].values, p_base_clean)

        # Mô hình tối ưu CafeRevenueModel
        model = CafeRevenueModel(**MODEL_PARAMS).fit(
            tr_split, stores_raw, weather_raw, holidays_raw
        )
        p_model_raw = model.predict(va_raw)
        p_model_clean = model.predict(va_clean)
        score_model_raw = rmsle(va_raw["revenue"].values, p_model_raw)
        score_model_clean = rmsle(va_clean["revenue"].values, p_model_clean)

        results.append(
            {
                "Quarter": q_name,
                "Baseline (Raw)": score_base_raw,
                "Baseline (Clean)": score_base_clean,
                "CafeModel (Raw)": score_model_raw,
                "CafeModel (Clean)": score_model_clean,
                "Improvement": (score_base_clean - score_model_clean)
                / score_base_clean
                * 100,
            }
        )

    res_df = pd.DataFrame(results)
    avg_row = {
        "Quarter": "Trung binh 4 quy",
        "Baseline (Raw)": res_df["Baseline (Raw)"].mean(),
        "Baseline (Clean)": res_df["Baseline (Clean)"].mean(),
        "CafeModel (Raw)": res_df["CafeModel (Raw)"].mean(),
        "CafeModel (Clean)": res_df["CafeModel (Clean)"].mean(),
        "Improvement": res_df["Improvement"].mean(),
    }
    res_df = pd.concat([res_df, pd.DataFrame([avg_row])], ignore_index=True)

    print("\n" + res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Huấn luyện mô hình toàn diện trên 100% dữ liệu lịch sử
    print("\n[3/4] Huan luyen mo hinh toan dien tren 100% du lieu (2024-2025)...")
    full_cafe_model = CafeRevenueModel(**MODEL_PARAMS).fit(
        train_raw, stores_raw, weather_raw, holidays_raw
    )
    full_baseline_model = BaselineWeekdayModel().fit(train_raw)

    # 4. Lưu mô hình vào models/
    print("\n[4/4] Luu cac mo hinh vao thu muc models/ ...")
    full_cafe_model.save(CAFE_MODEL_PATH)
    import joblib

    joblib.dump(full_baseline_model, BASELINE_MODEL_PATH)
    print(f"  + Da luu CafeRevenueModel vao: {CAFE_MODEL_PATH}")
    print(f"  + Da luu BaselineWeekdayModel vao: {BASELINE_MODEL_PATH}")
    print("\n>>> HOAN TAT HUAN LUYEN THANH CONG!")


if __name__ == "__main__":
    main()
