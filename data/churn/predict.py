# -*- coding: utf-8 -*-
"""
predict.py
Script sinh file nộp bài Kaggle (submission.csv) cho bài toán Card Churn Prediction.

Cách chạy:
    python predict.py

Nhiệm vụ:
    1. Tải mô hình đã huấn luyện từ models/churn_model.pkl (hoặc huấn luyện tự động nếu chưa có)
    2. Đọc tập test (test.csv nếu có, hoặc sử dụng sample_submission.csv khi chạy trên máy cá nhân)
    3. Dự báo xác suất churn_3m (từ 0.0 đến 1.0)
    4. Lưu kết quả vào submissions/submission.csv và submission.csv ở thư mục gốc
    5. Kiểm tra toàn vẹn dữ liệu (đúng 8.000 dòng, đúng cột customer_id và churn_3m, không có NaN).
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# Đảm bảo terminal console hiển thị UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    TRAIN_PATH,
    SAMPLE_SUB_PATH,
    TEST_PATH,
    CHURN_MODEL_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_BASELINE_PATH,
    MODEL_PARAMS,
)
from src.baseline_model import BaselineLogisticModel
from src.churn_model import ChurnPipeline


def main():
    print("=" * 80)
    print("SINH FILE SUBMISSION DU BAO CARD CHURN (KAGGLE)")
    print("=" * 80)

    # 1. Kiểm tra mô hình
    print("\n[1/4] Kiem tra & Tai mo hinh tu models/ ...")
    if os.path.exists(CHURN_MODEL_PATH) and os.path.exists(BASELINE_MODEL_PATH):
        print(f"  + Tim thay mo hinh co san: {CHURN_MODEL_PATH}")
        pipeline = ChurnPipeline.load(CHURN_MODEL_PATH)
        base_model = joblib.load(BASELINE_MODEL_PATH)
    else:
        print("  + Chua co mo hinh san, tien hanh huan luyen moi...")
        train_raw = pd.read_csv(TRAIN_PATH)
        pipeline = ChurnPipeline(**MODEL_PARAMS)
        pipeline.fit(train_raw)
        pipeline.save(CHURN_MODEL_PATH)

        base_model = BaselineLogisticModel()
        base_model.fit(train_raw)
        joblib.dump(base_model, BASELINE_MODEL_PATH)

    # 2. Đọc tập kiểm thử
    print("\n[2/4] Kiem tra tap kiem thu test.csv / sample_submission.csv ...")
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    has_test = os.path.exists(TEST_PATH)

    if has_test:
        print(f"  + Phat hien test.csv: {TEST_PATH}")
        test_df = pd.read_csv(TEST_PATH)
        pred_churn = pipeline.predict(test_df)
        pred_base = base_model.predict(test_df)
        sub_source = "test.csv (Du doan tu dac trung khach hang)"
    else:
        print("  + Chu y: May cuc bo chua co file test.csv (tren Kaggle file nay nam o /kaggle/input/...)")
        print("  + Tao submission chuan dinh dang tu sample_submission.csv voi xac suat mac dinh.")
        pred_churn = np.full(len(sample_sub), 0.1166)
        pred_base = np.full(len(sample_sub), 0.1200)
        sub_source = "sample_submission.csv (Khung mau)"

    # 3. Xuất file submission
    print("\n[3/4] Xuat file submission ...")
    sub_cafe = pd.DataFrame({
        "customer_id": sample_sub["customer_id"].values,
        "churn_3m": np.round(pred_churn, 6),
    })
    sub_base = pd.DataFrame({
        "customer_id": sample_sub["customer_id"].values,
        "churn_3m": np.round(pred_base, 6),
    })

    sub_cafe.to_csv(SUBMISSION_PATH, index=False)
    sub_base.to_csv(SUBMISSION_BASELINE_PATH, index=False)

    # Lưu thêm 1 bản tại thư mục gốc để nộp trực tiếp
    root_sub = os.path.join(os.path.dirname(SUBMISSION_PATH), "..", "submission.csv")
    sub_cafe.to_csv(root_sub, index=False)

    print(f"  + Da luu file toi uu: {SUBMISSION_PATH}")
    print(f"  + Da luu file baseline: {SUBMISSION_BASELINE_PATH}")

    # 4. Kiểm tra tính toàn vẹn
    print("\n" + "-" * 50)
    print("KIEM TRA TOAN VEN FILE SUBMISSION:")
    print(f"  + So luong dong: {len(sub_cafe):,} (Yeu cau: {len(sample_sub):,}) -> {'DUNG' if len(sub_cafe) == len(sample_sub) else 'SAI'}")
    print(f"  + Cac cot: {list(sub_cafe.columns)} -> {'DUNG' if list(sub_cafe.columns) == ['customer_id', 'churn_3m'] else 'SAI'}")
    print(f"  + Gia tri NaN: {sub_cafe['churn_3m'].isna().sum()} -> {'DUNG' if sub_cafe['churn_3m'].isna().sum() == 0 else 'SAI'}")
    print(f"  + Khoang xac suat: Min = {sub_cafe['churn_3m'].min():.4f}, Max = {sub_cafe['churn_3m'].max():.4f}")
    print(f"  + Nguon du doan: {sub_source}")
    print("-" * 50)
    print("\n>>> HOAN TAT TAO SUBMISSION THANH CONG!")


if __name__ == "__main__":
    main()
