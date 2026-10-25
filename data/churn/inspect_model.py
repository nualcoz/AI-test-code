# -*- coding: utf-8 -*-
"""
inspect_model.py
Script xem và khám phá các trọng số bên trong file models (.pkl) của bài toán Card Churn.

Cách dùng:
    python inspect_model.py
"""

import sys
import joblib
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import CHURN_MODEL_PATH, BASELINE_MODEL_PATH
from src.features import GAM_BIN_COLS, CAT_COLS, SPLINE_COLS


def inspect_baseline():
    print("=" * 80)
    print("1. NOI DUNG FILE: baseline_model.pkl (Mo Hinh Baseline Logistic Regression)")
    print("=" * 80)
    base = joblib.load(BASELINE_MODEL_PATH)
    print(f"Kieu du lieu: {type(base)}")
    print(f"So bien so dau vao (num_cols): {len(base.num_cols_)}")
    print("\nTrung vi dung de dien khuyet (fill_medians):")
    for col, med in base.fill_medians_.items():
        print(f"  + {col:25s}: {med}")

    lr = base.pipeline_.named_steps["logisticregression"]
    coefs = pd.Series(lr.coef_[0], index=base.num_cols_).sort_values(ascending=False)
    print("\nHe so hoi quy Logistic (Tren bien so chuan hoa):")
    for col, c in coefs.items():
        print(f"  + {col:25s}: {c:+.4f}")


def inspect_churn_model():
    print("\n" + "=" * 80)
    print("2. NOI DUNG FILE: churn_model.pkl (Mo Hinh AI ChurnPipeline Toi Uu)")
    print("=" * 80)
    pipe = joblib.load(CHURN_MODEL_PATH)
    cleaner = pipe.cleaner
    predictor = pipe.predictor

    print(f"Kieu du lieu: {type(pipe)}")
    print("\n[A] Thong so lam sach & suy luan cheo cua ChurnDataCleaner:")
    print("  + Ty le han muc / thu nhap theo loai the (card_limit_income_ratios):")
    for k, v in cleaner.card_limit_income_ratios_.items():
        print(f"     * {k:12s}: {v:.2f}x thu nhap")
    print(f"  + Ty le chi tieu giao dich / thu nhap (avg_txn_income_ratio): {cleaner.avg_txn_income_ratio_:.3f}")
    print("  + Trung vi tuoi theo nhom nghe nghiep:")
    for occ, age in cleaner.occ_age_medians_.items():
        print(f"     * {occ:15s}: {age:.1f} tuoi")

    print("\n[B] Kien truc Mo hinh Du bao (ChurnPredictor):")
    print(f"  + So luong cay Multi-seed HistGradientBoosting: {len(predictor.hgb_models_)} cay")
    print(f"  + Trong so nhung GAM Spline: {predictor.gam_weight*100:.0f}%")
    print(f"  + Trong so nhung GBDT: {(1-predictor.gam_weight)*100:.0f}%")

    print("\n[C] He so LogisticRegression trong nhanh Spline GAM (Nghiep vu chu chot):")
    gam_lr = predictor.gam_model_.named_steps["logisticregression"]
    # Trích xuất tên các cột trong GAM
    ct = predictor.gam_model_.named_steps["columntransformer"]
    sp_names = [f"spline_{i}" for i in range(len(SPLINE_COLS) * 10)]
    bin_names = GAM_BIN_COLS
    cat_enc = ct.named_transformers_["cat"]
    cat_names = list(cat_enc.get_feature_names_out(CAT_COLS))
    all_names = sp_names + bin_names + cat_names

    gam_coefs = pd.Series(gam_lr.coef_[0][: len(all_names)], index=all_names)
    print("  + Tac dong cac yeu to rui ro chu chot (Odds Ratio):")
    for col in bin_names:
        if col in gam_coefs:
            val = gam_coefs[col]
            odds = np.exp(val)
            print(f"     * {col:25s}: he so = {val:+.4f} (Ty le nguy co Odds x{odds:.2f})")


def main():
    inspect_baseline()
    inspect_churn_model()
    print("\n" + "=" * 80)
    print(">>> KET THUC XEM NOI DUNG MODEL CHURN")
    print("=" * 80)


if __name__ == "__main__":
    main()
