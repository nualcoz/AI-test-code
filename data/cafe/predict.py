"""
predict.py
Script sinh file nộp bài Kaggle (submission.csv) cho bài toán Cafe Revenue Prediction.

Cách chạy:
    python predict.py

Nhiệm vụ:
    1. Tải mô hình đã huấn luyện từ models/cafe_revenue_model.pkl (hoặc huấn luyện tự động nếu chưa có)
    2. Chuẩn bị tập kiểm thử Q1/2026 (3.869 dòng) khớp chính xác với sample_submission.csv
    3. Dự báo doanh thu và xuất file ra submissions/submission.csv
    4. Kiểm tra toàn vẹn định dạng file nộp bài (3.869 dòng, cột id và revenue, không chứa giá trị NaN).
"""

import os
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
    SAMPLE_SUB_PATH,
    CAFE_MODEL_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_BASELINE_PATH,
    MODEL_PARAMS,
)
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model import CafeRevenueModel, build_test_grid_if_missing


def main():
    print("=" * 80)
    print("SINH FILE SUBMISSION DU BAO DOANH THU QUY 1/2026 (KAGGLE)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    print("\n[1/4] Dang doc cac bang du lieu tu data/ ...")
    stores_raw = pd.read_csv(STORES_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    print(f"  + sample_submission.csv: {sample_sub.shape[0]:,} dong can du bao")

    # 2. Tải hoặc huấn luyện mô hình
    print("\n[2/4] Tai mo hinh tu models/ ...")
    if os.path.exists(CAFE_MODEL_PATH) and os.path.exists(BASELINE_MODEL_PATH):
        print(f"  + Tim thay mo hinh co san: {CAFE_MODEL_PATH}")
        model = CafeRevenueModel.load(CAFE_MODEL_PATH)
        import joblib

        base_model = joblib.load(BASELINE_MODEL_PATH)
    else:
        print("  + Chua co mo hinh san, tien hanh huan luyen moi...")
        train_raw = pd.read_csv(TRAIN_PATH)
        weather_raw = pd.read_csv(WEATHER_PATH)
        holidays_raw = pd.read_csv(HOLIDAYS_PATH)
        model = CafeRevenueModel(**MODEL_PARAMS).fit(
            train_raw, stores_raw, weather_raw, holidays_raw
        )
        base_model = BaselineWeekdayModel().fit(train_raw)
        model.save(CAFE_MODEL_PATH)
        import joblib

        joblib.dump(base_model, BASELINE_MODEL_PATH)

    # 3. Chuẩn bị tập test Q1/2026
    print("\n[3/4] Chuan bi tap du bao Q1/2026 (01/01/2026 - 31/03/2026)...")
    test_df = build_test_grid_if_missing(stores_raw, sample_sub)
    print(f"  + So dong can du bao: {len(test_df):,} dong (Khop voi sample_submission)")

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

    sub_cafe.to_csv(SUBMISSION_PATH, index=False)
    sub_base.to_csv(SUBMISSION_BASELINE_PATH, index=False)

    # Đảm bảo file được copy vào thư mục gốc để người dùng nộp trực tiếp nếu muốn
    root_sub_cafe = os.path.join(os.path.dirname(SUBMISSION_PATH), "..", "submission.csv")
    sub_cafe.to_csv(root_sub_cafe, index=False)

    print(f"  + Da luu file toi uu: {SUBMISSION_PATH}")
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
    print("\n>>> HOAN TAT TAO SUBMISSION THANH CONG!")


if __name__ == "__main__":
    main()
