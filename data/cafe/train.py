"""
train.py
Script huấn luyện và kiểm định chéo mô hình dự báo doanh thu chuỗi quán Cafe.

Cách chạy:
    python train.py

Nhiệm vụ:
    1. Đọc dữ liệu từ data/ (train.csv, test.csv, stores.csv, weather.csv, holidays.csv)
    2. Làm sạch dữ liệu toàn diện (khử nhiễu 1000x, 10x, 0.1x, F->C, điền diện tích)
    3. Đánh giá kiểm định chéo qua 4 quý năm 2025 (Q1, Q2, Q3, Q4) trên các mô hình
    4. Huấn luyện mô hình Thế hệ 3 (CafeRevenueModelV3) trên 100% dữ liệu lịch sử
    5. Lưu trọng số mô hình vào models/
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

# Thiết lập encoding UTF-8 cho console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    STORES_PATH,
    WEATHER_PATH,
    HOLIDAYS_PATH,
    CAFE_MODEL_PATH,
    CAFE_MODEL_V2_PATH,
    CAFE_MODEL_V3_PATH,
    BASELINE_MODEL_PATH,
    MODEL_PARAMS,
    EVAL_QUARTERS,
)
from src.cleaning import clean_train, clean_stores, clean_test
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model import CafeRevenueModel
from src.cafe_model_v2 import CafeRevenueModelV2
from src.cafe_model_v3 import CafeRevenueModelV3


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính Root Mean Squared Logarithmic Error."""
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def main():
    print("=" * 80)
    print("HUAN LUYEN & KIEM DINH CHEO MO HINH DU BAO DOANH THU CAFE (THE HE 3)")
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
    
    if os.path.exists(TEST_PATH):
        test_raw = pd.read_csv(TEST_PATH)
        print(f"  + test.csv: {test_raw.shape[0]:,} dong can du bao")
    else:
        print("  ! Canh bao: Khong tim thay test.csv trong data/")

    # 2. Đánh giá kiểm định chéo theo quý (Out-of-Time Quarter Validation)
    print("\n[2/4] Danh gia Kiem dinh cheo theo 4 quy nam 2025...")
    stores_c = clean_stores(stores_raw)
    train_c_all, stats = clean_train(train_raw, stores_c)
    print(f"  + Thong ke lam sach du lieu train: {stats}")

    results = []
    for q_name, (val_start, val_end) in EVAL_QUARTERS.items():
        tr_split = train_raw[train_raw["date"] < val_start].copy()
        va_clean = train_c_all[
            (train_c_all["date"] >= val_start) & (train_c_all["date"] <= val_end)
        ].copy()

        # Baseline gốc
        base = BaselineWeekdayModel().fit(tr_split)
        p_base = base.predict(va_clean)
        score_base = rmsle(va_clean["revenue"].values, p_base)

        # Mô hình CafeModel V1
        m_v1 = CafeRevenueModel(**MODEL_PARAMS).fit(
            tr_split, stores_raw, weather_raw, holidays_raw
        )
        p_v1 = m_v1.predict(va_clean)
        score_v1 = rmsle(va_clean["revenue"].values, p_v1)

        # Mô hình CafeModel V3 (Toi uu hoa)
        m_v3 = CafeRevenueModelV3().fit(
            tr_split, stores_raw, weather_raw, holidays_raw
        )
        p_v3 = m_v3.predict(va_clean)
        score_v3 = rmsle(va_clean["revenue"].values, p_v3)

        results.append(
            {
                "Quarter": q_name,
                "Baseline": score_base,
                "CafeModel_v1": score_v1,
                "CafeModel_v3": score_v3,
                "Improvement_vs_Base": (score_base - score_v3) / score_base * 100,
            }
        )

    res_df = pd.DataFrame(results)
    avg_row = {
        "Quarter": "Trung binh 4 quy",
        "Baseline": res_df["Baseline"].mean(),
        "CafeModel_v1": res_df["CafeModel_v1"].mean(),
        "CafeModel_v3": res_df["CafeModel_v3"].mean(),
        "Improvement_vs_Base": res_df["Improvement_vs_Base"].mean(),
    }
    res_df = pd.concat([res_df, pd.DataFrame([avg_row])], ignore_index=True)

    print("\n" + res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Huấn luyện mô hình toàn diện trên 100% dữ liệu lịch sử
    print("\n[3/4] Huan luyen mo hinh toan dien tren 100% du lieu (2024-2025)...")
    full_m_v3 = CafeRevenueModelV3().fit(
        train_raw, stores_raw, weather_raw, holidays_raw
    )
    full_baseline_model = BaselineWeekdayModel().fit(train_raw)

    # 4. Lưu mô hình vào models/
    print("\n[4/4] Luu cac mo hinh vao thu muc models/ ...")
    full_m_v3.save(CAFE_MODEL_V3_PATH)
    joblib.dump(full_baseline_model, BASELINE_MODEL_PATH)
    print(f"  + Da luu CafeRevenueModelV3 vao: {CAFE_MODEL_V3_PATH}")
    print(f"  + Da luu BaselineWeekdayModel vao: {BASELINE_MODEL_PATH}")
    print("\n>>> HOAN TAT HUAN LUYEN THANH CONG!")


if __name__ == "__main__":
    main()
