# -*- coding: utf-8 -*-
"""
train.py
Script huấn luyện và kiểm thử mô hình dự báo xác suất rời bỏ thẻ tín dụng (Card Churn).

Cách chạy:
    python train.py

Nhiệm vụ:
    1. Đọc dữ liệu từ data/train.csv
    2. Kiểm toán và làm sạch dữ liệu (sửa lỗi 1e6, lọc giá trị sentinel, suy luận chéo)
    3. Đánh giá kiểm định chéo 5-Fold Stratified K-Fold (ROC-AUC) qua từng giai đoạn cải tiến
    4. Huấn luyện toàn diện ChurnPipeline trên 100% dữ liệu
    5. Lưu các mô hình vào models/churn_model.pkl và models/baseline_model.pkl
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Đảm bảo terminal console hiển thị UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    TRAIN_PATH,
    CHURN_MODEL_PATH,
    BASELINE_MODEL_PATH,
    MODEL_PARAMS,
    CV_CONFIG,
)
from src.cleaning import ChurnDataCleaner
from src.baseline_model import BaselineLogisticModel
from src.churn_model import ChurnPredictor, ChurnPipeline


def evaluate_cv(train_raw: pd.DataFrame, train_clean: pd.DataFrame):
    """Đánh giá 5-Fold Stratified K-Fold CV qua các mốc phát triển."""
    cv = StratifiedKFold(
        n_splits=CV_CONFIG["n_splits"],
        shuffle=CV_CONFIG["shuffle"],
        random_state=CV_CONFIG["random_state"],
    )

    # 1. Baseline trên dữ liệu thô (đã bỏ trùng để công bằng)
    raw_dedup = train_raw.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)
    raw_num_cols = [c for c in raw_dedup.select_dtypes("number").columns if c != "churn_3m"]
    X_raw = raw_dedup[raw_num_cols].fillna(raw_dedup[raw_num_cols].median())
    y = train_clean["churn_3m"].values

    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))

    clean_num_cols = [
        "age_imputed",
        "income_imputed",
        "credit_limit",
        "annual_fee_waived",
        "utilization_imputed",
        "txn_count_3m",
        "txn_amt_imputed",
        "spend_change_pct",
        "app_logins_imputed",
        "complaints_12m",
        "late_payments_12m",
        "other_cards_imputed",
        "promo_ended_60d",
    ]

    oof_base = np.zeros(len(train_clean))
    oof_clean_lr = np.zeros(len(train_clean))
    oof_gam = np.zeros(len(train_clean))
    oof_hgb = np.zeros(len(train_clean))
    oof_ens = np.zeros(len(train_clean))

    fold_rows = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(train_clean, y), start=1):
        # Baseline Fold
        lr_pipe.fit(X_raw.iloc[tr_idx], y[tr_idx])
        oof_base[va_idx] = lr_pipe.predict_proba(X_raw.iloc[va_idx])[:, 1]

        # Cleaned Numeric LR Fold
        lr_pipe.fit(train_clean.iloc[tr_idx][clean_num_cols], y[tr_idx])
        oof_clean_lr[va_idx] = lr_pipe.predict_proba(train_clean.iloc[va_idx][clean_num_cols])[:, 1]

        # ChurnPredictor (Spline GAM + Multi-seed HGB)
        pred = ChurnPredictor(**MODEL_PARAMS)
        pred.fit(train_clean.iloc[tr_idx], y[tr_idx])
        p_hgb, p_gam, p_ens = pred.predict_proba_components(train_clean.iloc[va_idx])

        oof_hgb[va_idx] = p_hgb
        oof_gam[va_idx] = p_gam
        oof_ens[va_idx] = p_ens

        fold_rows.append({
            "Fold": f"Fold {fold}",
            "Baseline (Thô)": roc_auc_score(y[va_idx], oof_base[va_idx]),
            "Clean Numeric": roc_auc_score(y[va_idx], oof_clean_lr[va_idx]),
            "Spline GAM": roc_auc_score(y[va_idx], p_gam),
            "MultiSeed HGB": roc_auc_score(y[va_idx], p_hgb),
            "Hybrid Ensemble": roc_auc_score(y[va_idx], p_ens),
        })

    avg_row = {
        "Fold": "OOF Trung Binh",
        "Baseline (Thô)": roc_auc_score(y, oof_base),
        "Clean Numeric": roc_auc_score(y, oof_clean_lr),
        "Spline GAM": roc_auc_score(y, oof_gam),
        "MultiSeed HGB": roc_auc_score(y, oof_hgb),
        "Hybrid Ensemble": roc_auc_score(y, oof_ens),
    }
    fold_df = pd.concat([pd.DataFrame(fold_rows), pd.DataFrame([avg_row])], ignore_index=True)
    return fold_df


def main():
    print("=" * 80)
    print("HUAN LUYEN & KIEM THU MO HINH DU BAO CARD CHURN (ROC-AUC)")
    print("=" * 80)

    # 1. Đọc dữ liệu
    print("\n[1/4] Dang doc du lieu tu data/train.csv ...")
    train_raw = pd.read_csv(TRAIN_PATH)
    print(f"  + Tong so dong ban dau: {train_raw.shape[0]:,} dong")
    print(f"  + Tong so cot: {train_raw.shape[1]} cot")
    print(f"  + Ty le churn goc: {train_raw['churn_3m'].mean()*100:.2f}%")

    # 2. Làm sạch và kiểm toán lỗi
    print("\n[2/4] Lam sach du lieu & Kiem toan cac bat thuong ...")
    cleaner = ChurnDataCleaner()
    train_clean, errors_found = cleaner.clean(train_raw, deduplicate=True)
    print(f"  + So dong sau khi loai bo trung lap: {len(train_clean):,} dong")
    print(f"  + Tong so loi bat thuong da phat hien & sua chua: {len(errors_found):,} loi")
    print("  + Thong ke chi tiet theo loai loi:")
    for err_type, count in errors_found["error_type"].value_counts().items():
        print(f"     * {err_type:30s}: {count:4d} dong")

    # 3. Đánh giá 5-Fold Stratified Cross-Validation
    print("\n[3/4] Danh gia Kiem dinh cheo 5-Fold Stratified CV (ROC-AUC)...")
    cv_results = evaluate_cv(train_raw, train_clean)
    print("\n" + cv_results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 4. Huấn luyện toàn diện trên 100% dữ liệu sạch & Lưu mô hình
    print("\n[4/4] Huan luyen mo hinh toan dien tren 100% du lieu & Luu file ...")
    full_pipeline = ChurnPipeline(**MODEL_PARAMS)
    full_pipeline.fit(train_raw)
    full_pipeline.save(CHURN_MODEL_PATH)

    full_baseline = BaselineLogisticModel()
    full_baseline.fit(train_raw)
    import joblib

    joblib.dump(full_baseline, BASELINE_MODEL_PATH)

    print(f"  + Da luu ChurnPipeline vao: {CHURN_MODEL_PATH}")
    print(f"  + Da luu BaselineLogisticModel vao: {BASELINE_MODEL_PATH}")
    print("\n>>> HOAN TAT HUAN LUYEN THANH CONG!")


if __name__ == "__main__":
    main()
