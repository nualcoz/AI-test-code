"""
predict.py
Script sinh file nộp bài Kaggle (submission.csv) cho bài toán Cafe Revenue Prediction.
Sử dụng trực tiếp tập kiểm thử thực tế test.csv và mô hình CafeRevenueModelV3.

Cách chạy:
    python predict.py

Nhiệm vụ:
    1. Đọc test.csv trực tiếp từ data/test.csv và làm sạch/kiểm toán định dạng
    2. Tải mô hình đã huấn luyện từ models/cafe_revenue_model_v3.pkl (hoặc huấn luyện tự động nếu chưa có)
    3. Dự báo doanh thu với đặc trưng khuyến mãi is_promo thực tế
    4. Xuất file submission ra submissions/submission_v3.csv và submission.csv ở thư mục gốc
    5. Kiểm tra toàn vẹn định dạng file nộp bài (3.869 dòng, cột id và revenue, không chứa giá trị NaN).
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
    SAMPLE_SUB_PATH,
    CAFE_MODEL_V3_PATH,
    CAFE_MODEL_V2_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_V3_PATH,
    SUBMISSION_BASELINE_PATH,
)
from src.cleaning import clean_stores, clean_test
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model_v3 import CafeRevenueModelV3
from src.cafe_model import build_test_grid_if_missing


def main():
    print("=" * 80)
    print("SINH FILE SUBMISSION DU BAO DOANH THU QUY 1/2026 (KAGGLE)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    print("\n[1/4] Dang doc cac bang du lieu tu data/ ...")
    stores_raw = pd.read_csv(STORES_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    stores_c = clean_stores(stores_raw)
    print(f"  + sample_submission.csv: {sample_sub.shape[0]:,} dong can du bao")

    # 2. Đọc và kiểm toán test.csv
    print("\n[2/4] Doc va kiem toan tap kiem thu test.csv ...")
    if os.path.exists(TEST_PATH):
        test_raw = pd.read_csv(TEST_PATH)
        test_df, test_stats = clean_test(test_raw, stores_c)
        print(f"  + Tim thay test.csv voi {len(test_df):,} dong")
        print(f"  + So ngay khuyen mai thuc te: {test_stats['promo_days_count']} ngay ({test_stats['promo_ratio']*100:.1f}%)")
    else:
        print("  ! Khong tim thay test.csv, su dung luoi gia dinh tu build_test_grid_if_missing...")
        test_df = build_test_grid_if_missing(stores_raw, sample_sub)

    # 3. Tải hoặc huấn luyện mô hình
    print("\n[3/4] Tai mo hinh tu models/ ...")
    if os.path.exists(CAFE_MODEL_V3_PATH):
        print(f"  + Tim thay mo hinh co san: {CAFE_MODEL_V3_PATH}")
        model = CafeRevenueModelV3.load(CAFE_MODEL_V3_PATH)
    elif os.path.exists(CAFE_MODEL_V2_PATH):
        print(f"  + Tim thay mo hinh V2: {CAFE_MODEL_V2_PATH}")
        from src.cafe_model_v2 import CafeRevenueModelV2
        model = CafeRevenueModelV2.load(CAFE_MODEL_V2_PATH)
    else:
        print("  + Chua co mo hinh san, tien hanh huan luyen moi CafeRevenueModelV3...")
        train_raw = pd.read_csv(TRAIN_PATH)
        weather_raw = pd.read_csv(WEATHER_PATH)
        holidays_raw = pd.read_csv(HOLIDAYS_PATH)
        model = CafeRevenueModelV3().fit(train_raw, stores_raw, weather_raw, holidays_raw)
        model.save(CAFE_MODEL_V3_PATH)

    if os.path.exists(BASELINE_MODEL_PATH):
        base_model = joblib.load(BASELINE_MODEL_PATH)
    else:
        train_raw = pd.read_csv(TRAIN_PATH)
        base_model = BaselineWeekdayModel().fit(train_raw)
        joblib.dump(base_model, BASELINE_MODEL_PATH)

    # 4. Dự báo và xuất file submission
    print("\n[4/4] Du bao doanh thu va xuat file ...")
    pred_cafe = model.predict(test_df)
    pred_base = base_model.predict(test_df)

    sub_cafe = pd.DataFrame({
        "id": test_df["id"].values,
        "revenue": np.round(pred_cafe, 4),
    })
    sub_base = pd.DataFrame({
        "id": test_df["id"].values,
        "revenue": np.round(pred_base, 4),
    })

    # Lưu cả submission.csv gốc và submission_v3.csv
    sub_cafe.to_csv(SUBMISSION_V3_PATH, index=False)
    sub_cafe.to_csv(SUBMISSION_PATH, index=False)
    sub_base.to_csv(SUBMISSION_BASELINE_PATH, index=False)

    # Đảm bảo file được copy vào thư mục gốc để người dùng nộp trực tiếp nếu muốn
    root_sub_cafe = os.path.join(os.path.dirname(SUBMISSION_PATH), "..", "submission.csv")
    sub_cafe.to_csv(root_sub_cafe, index=False)

    print(f"  + Da luu file toi uu: {SUBMISSION_V3_PATH}")
    print(f"  + Da luu file goc: {root_sub_cafe}")
    print(f"  + Da luu file baseline: {SUBMISSION_BASELINE_PATH}")

    # Kiểm tra tính toàn vẹn
    print("\n" + "-" * 50)
    print("KIEM TRA TOAN VEN FILE SUBMISSION:")
    print(f"  + So luong dong: {len(sub_cafe)} (Yeu cau: {len(sample_sub)}) -> {'DUNG' if len(sub_cafe) == len(sample_sub) else 'SAI'}")
    print(f"  + Cac cot: {list(sub_cafe.columns)} -> {'DUNG' if list(sub_cafe.columns) == ['id', 'revenue'] else 'SAI'}")
    print(f"  + Gia tri NaN: {sub_cafe['revenue'].isna().sum()} -> {'DUNG' if sub_cafe['revenue'].isna().sum() == 0 else 'SAI'}")
    print(f"  + Gia tri < 0: {(sub_cafe['revenue'] < 0).sum()} -> {'DUNG' if (sub_cafe['revenue'] < 0).sum() == 0 else 'SAI'}")
    print(f"  + Doanh thu du bao: Min = {sub_cafe['revenue'].min():.2f} tr, Median = {sub_cafe['revenue'].median():.2f} tr, Max = {sub_cafe['revenue'].max():.2f} tr")
    print("-" * 50)
    print("\n5 dong dau tien cua file submission:")
    print(sub_cafe.head().to_string(index=False))
    print("\n>>> HOAN TAT TAO SUBMISSION THANH CONG!")


if __name__ == "__main__":
    main()
