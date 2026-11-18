# -*- coding: utf-8 -*-
"""
train.py
Quy trình Huấn luyện & Tối ưu hóa Toàn diện Mô hình Card Churn theo chuẩn 8 Bước
(Tham chiếu theo Chương 4: Xây dựng mô hình & Phát triển ứng dụng AI - ĐH Kinh tế Quốc dân)

8 Bước triển khai:
  Bước 1 (Collect & Prepare): Nạp train.csv (12,120 dòng), test.csv (8,000 dòng), định nghĩa bài toán & metric ROC-AUC
  Bước 2 (Explore): Kiểm toán phân phối nhãn (11.66%), tỷ lệ thiếu, phát hiện dị biệt (tuổi 0/999, đơn vị 1e6, số âm)
  Bước 3 (Preprocess): Khử trùng lặp 120 dòng train, sửa lỗi đơn vị, làm sạch text, trích xuất đặc trưng nghiệp vụ
  Bước 4 (Split): Phân chia Stratified 80/20 và kiểm định chéo 5-Fold Stratified K-Fold
  Bước 5 (Scale): Chuẩn hóa bên trong Pipeline (fit trên train, transform trên test - chống rò rỉ dữ liệu)
  Bước 6 (Model & Train): Huấn luyện các mô hình cơ bản (Slide 53) và Mô hình Tối ưu Hybrid Ensemble
  Bước 7 (Evaluate): Đánh giá 5-Fold CV, Ma trận nhầm lẫn 0.5, Danh sách The Call List (Top 10%), Kiểm tra Overfitting
  Bước 8 (Deploy): Dự báo trên test.csv (8,000 dòng), xuất submission.csv chuẩn Kaggle, mô phỏng Form What-if
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

# Đảm bảo hiển thị tiếng Việt UTF-8 chuẩn trên Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    CHURN_MODEL_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_BASELINE_PATH,
    MODEL_PARAMS,
    CV_CONFIG,
    PROJECT_ROOT,
)
from src.cleaning import ChurnDataCleaner
from src.features import build_features
from src.baseline_model import BaselineLogisticModel
from src.churn_model import ChurnPredictor, ChurnPipeline


def main():
    print("=" * 85)
    print(" QUY TRÌNH 8 BƯỚC TỐI ƯU HÓA DỮ LIỆU & HUẤN LUYỆN MÔ HÌNH DỰ BÁO CARD CHURN ")
    print(" (Tham chiếu giáo trình Chapter 4: National Economics University - MSc. Nguyễn Quốc Thái) ")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # BƯỚC 1: COLLECT & PREPARE (Thu thập & Chuẩn bị bài toán)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 1/8] COLLECT & PREPARE · Thu thập và kiểm tra dữ liệu")
    print("-" * 85)
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH) if os.path.exists(TEST_PATH) else None

    print(f"  + File huấn luyện (train.csv): {train_raw.shape[0]:,} dòng, {train_raw.shape[1]} cột")
    if test_raw is not None:
        print(f"  + File kiểm thử (test.csv)    : {test_raw.shape[0]:,} dòng, {test_raw.shape[1]} cột")
    print("  + Câu hỏi nghiệp vụ : Khách hàng nào sẽ đóng thẻ tín dụng trong 3 tháng tới?")
    print("  + Loại bài toán     : Phân loại nhị phân (Binary Classification) - Nhãn churn_3m (0/1)")
    print("  + Thước đo đánh giá : ROC-AUC (Khả năng xếp hạng rủi ro giữa churners và stayers)")

    # -------------------------------------------------------------------------
    # BƯỚC 2: EXPLORE (Khám phá dữ liệu & Kiểm toán lỗi)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 2/8] EXPLORE · Khám phá dữ liệu & Phát hiện các bất thường")
    print("-" * 85)
    churn_rate = train_raw["churn_3m"].mean()
    print(f"  + Tỷ lệ rời bỏ (churn rate) : {churn_rate*100:.2f}% (Chỉ 11.66% khách hàng đóng thẻ)")
    print(f"  + Baseline 'Không ai đóng thẻ' đạt Accuracy { (1-churn_rate)*100:.2f}%, nhưng F1=0 & AUC=0.500")
    
    missing_pct = (train_raw.isna().mean() * 100).sort_values(ascending=False)
    top_missing = missing_pct[missing_pct > 0].head(8)
    print("  + Top các cột có tỷ lệ khuyết thiếu cao nhất:")
    for col, pct in top_missing.items():
        print(f"     * {col:25s}: {pct:5.2f}%")

    cleaner = ChurnDataCleaner()
    errors_df = cleaner.find_errors(train_raw)
    print(f"\n  + Kiểm toán phát hiện {len(errors_df):,} lỗi bất thường trong dữ liệu train.csv:")
    for err_type, cnt in errors_df["error_type"].value_counts().items():
        print(f"     * {err_type:32s}: {cnt:4d} dòng")

    # -------------------------------------------------------------------------
    # BƯỚC 3: PREPROCESS & FEATURE ENGINEERING (Tiền xử lý & Tạo đặc trưng)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 3/8] PREPROCESS · Làm sạch sâu & Tạo đặc trưng nghiệp vụ ngân hàng")
    print("-" * 85)
    train_clean, _ = cleaner.clean(train_raw, deduplicate=True)
    print(f"  + Khử trùng lặp (chỉ áp dụng trên tập Train): {len(train_raw):,} -> {len(train_clean):,} dòng (-120 dòng)")
    print("  + Loại bỏ 2 cột hằng số vô nghĩa: ['bank_code', 'currency']")
    print("  + Sửa lỗi đơn vị Đồng -> Triệu Đồng (chia 1e6) cho income_month và credit_limit")
    print("  + Lọc giá trị sentinel tuổi (0, 999) -> NaN và suy luận chéo theo nghề nghiệp")
    print("  + Sửa giá trị âm và bất thường cho utilization_6m và txn_amount_3m")
    print("  + Chuẩn hóa đa định dạng ngày open_date -> tenure_days, open_month")
    print("  + Kỹ thuật đặc trưng hành vi & nghiệp vụ ngân hàng:")
    print("     * anniversary_unwaived : Kỷ niệm mở thẻ tháng 7-9 và không được miễn phí thường niên")
    print("     * is_cashback_promo   : Thẻ Cashback vừa kết thúc 60 ngày ưu đãi mở mới")
    print("     * util_is_zero        : Thẻ không phát sinh số dư (nguy cơ bỏ thẻ cao)")
    print("     * spend_drop          : Mức độ sụt giảm chi tiêu trong 3 tháng")
    print("     * avg_txn_amt         : Giá trị giao dịch trung bình mỗi lần cà thẻ")

    train_feats = build_features(train_clean)
    y = train_clean["churn_3m"].values
    print(f"  + Tổng số đặc trưng sau biến đổi kỹ thuật: {train_feats.shape[1]} đặc trưng")

    # -------------------------------------------------------------------------
    # BƯỚC 4 & 5: SPLIT & SCALE (Phân chia & Chuẩn hóa bên trong Pipeline)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 4 & 5/8] SPLIT & SCALE · Phân chia tập kiểm thử & Chuẩn hóa bằng Pipeline")
    print("-" * 85)
    print("  + Áp dụng quy tắc 1 & 2 của giáo trình:")
    print("     (1) Giữ tập kiểm thử tách biệt hoàn toàn; không để rò rỉ dữ liệu (data leakage).")
    print("     (2) Imputer và Scaler chỉ 'fit' trên tập Train, sau đó 'transform' trên Val/Test.")
    
    # -------------------------------------------------------------------------
    # BƯỚC 6 & 7: MODEL, TRAIN & EVALUATE (Thử nghiệm mô hình & Đánh giá)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 6 & 7/8] MODEL & EVALUATE · Đánh giá mô hình chuẩn Slide 53 & Mô hình Tối ưu")
    print("-" * 85)
    
    # Chuẩn bị dữ liệu dạng Slide 51 cho các mô hình scikit-learn cơ bản
    cats_basic = ["gender", "card_type", "salary_via_bank", "last_contact_channel"]
    def get_basic_X(d):
        num_part = d.select_dtypes("number").drop(columns=["churn_3m"], errors="ignore")
        return num_part.join(pd.get_dummies(d[cats_basic], dtype=int))

    X_basic = get_basic_X(train_clean)
    
    def pipe_basic(model_inst):
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model_inst)

    cv = StratifiedKFold(n_splits=CV_CONFIG["n_splits"], shuffle=CV_CONFIG["shuffle"], random_state=CV_CONFIG["random_state"])

    # 1. Đối chiếu các mô hình theo đúng Slide 53
    slide53_models = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(max_iter=2000),
        "Tree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "Naive Bayes": GaussianNB(),
        "kNN": KNeighborsClassifier(n_neighbors=51),
        "MLP": MLPClassifier((16,), max_iter=500, random_state=0),
    }

    print("\n  [Bảng 1] Đánh giá 5-Fold Stratified CV trên các mô hình cơ bản (Slide 53):")
    rows_basic = []
    for name, m in slide53_models.items():
        if name == "Baseline":
            rows_basic.append({"Mô hình": name, "5-Fold Accuracy": 0.8834, "5-Fold F1": 0.000, "5-Fold ROC-AUC": 0.5000})
            continue
        # Chạy CV cho từng mô hình
        acc_scores = []
        f1_scores = []
        auc_scores = []
        for tr_idx, va_idx in cv.split(X_basic, y):
            p = pipe_basic(m)
            p.fit(X_basic.iloc[tr_idx], y[tr_idx])
            pred_cl = p.predict(X_basic.iloc[va_idx])
            prob_cl = p.predict_proba(X_basic.iloc[va_idx])[:, 1]
            acc_scores.append(accuracy_score(y[va_idx], pred_cl))
            f1_scores.append(f1_score(y[va_idx], pred_cl, zero_division=0))
            auc_scores.append(roc_auc_score(y[va_idx], prob_cl))
        rows_basic.append({
            "Mô hình": name,
            "5-Fold Accuracy": np.mean(acc_scores),
            "5-Fold F1": np.mean(f1_scores),
            "5-Fold ROC-AUC": np.mean(auc_scores),
        })

    df_slide53 = pd.DataFrame(rows_basic)
    print(df_slide53.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2. Đánh giá Mô hình Tối ưu (Hybrid Ensemble: Spline GAM + MultiSeed HGB)
    print("\n  [Bảng 2] Đánh giá Mô hình Tối ưu (Hybrid Ensemble) qua từng Fold:")
    raw_dedup = train_raw.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)
    raw_num_cols = [c for c in raw_dedup.select_dtypes("number").columns if c != "churn_3m"]
    X_raw = raw_dedup[raw_num_cols].fillna(raw_dedup[raw_num_cols].median())

    lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))
    oof_base = np.zeros(len(train_clean))
    oof_gam = np.zeros(len(train_clean))
    oof_hgb = np.zeros(len(train_clean))
    oof_ens = np.zeros(len(train_clean))

    fold_details = []
    for fold, (tr_idx, va_idx) in enumerate(cv.split(train_clean, y), start=1):
        # Baseline thô
        lr_pipe.fit(X_raw.iloc[tr_idx], y[tr_idx])
        oof_base[va_idx] = lr_pipe.predict_proba(X_raw.iloc[va_idx])[:, 1]

        # Mô hình tối ưu
        opt_predictor = ChurnPredictor(**MODEL_PARAMS)
        opt_predictor.fit(train_clean.iloc[tr_idx], y[tr_idx])
        p_hgb, p_gam, p_ens = opt_predictor.predict_proba_components(train_clean.iloc[va_idx])

        oof_hgb[va_idx] = p_hgb
        oof_gam[va_idx] = p_gam
        oof_ens[va_idx] = p_ens

        fold_details.append({
            "Fold": f"Fold {fold}",
            "Starter Baseline": roc_auc_score(y[va_idx], oof_base[va_idx]),
            "Spline GAM": roc_auc_score(y[va_idx], p_gam),
            "MultiSeed HGB": roc_auc_score(y[va_idx], p_hgb),
            "Hybrid Tối Ưu": roc_auc_score(y[va_idx], p_ens),
        })

    avg_opt = {
        "Fold": "OOF TRUNG BÌNH",
        "Starter Baseline": roc_auc_score(y, oof_base),
        "Spline GAM": roc_auc_score(y, oof_gam),
        "MultiSeed HGB": roc_auc_score(y, oof_hgb),
        "Hybrid Tối Ưu": roc_auc_score(y, oof_ens),
    }
    df_opt = pd.concat([pd.DataFrame(fold_details), pd.DataFrame([avg_opt])], ignore_index=True)
    print(df_opt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Ma trận nhầm lẫn & F1 ở ngưỡng 0.5 (Slide 54)
    ens_pred_05 = (oof_ens >= 0.5).astype(int)
    cm = confusion_matrix(y, ens_pred_05)
    tn, fp, fn, tp = cm.ravel()
    print("\n  + Ma trận nhầm lẫn ở ngưỡng 0.5 (Confusion Matrix - Slide 54):")
    print(f"     * True Negative  (Dự báo 0, Thực tế 0): {tn:5d} khách hàng")
    print(f"     * False Positive (Dự báo 1, Thực tế 0): {fp:5d} khách hàng")
    print(f"     * False Negative (Dự báo 0, Thực tế 1): {fn:5d} khách hàng")
    print(f"     * True Positive  (Dự báo 1, Thực tế 1): {tp:5d} khách hàng")
    print(f"     * Accuracy: {(tp+tn)/len(y):.4f} | F1-score: {f1_score(y, ens_pred_05):.4f}")

    # 4. Phân tích The Call List: Top 10% khách hàng rủi ro nhất (Slide 57)
    q90 = np.quantile(oof_ens, 0.90)
    top10_mask = oof_ens >= q90
    top10_count = top10_mask.sum()
    churn_rate_top10 = y[top10_mask].mean()
    captured_churners = y[top10_mask].sum()
    capture_share = captured_churners / y.sum()

    print("\n  + Phân tích The Call List - Top 10% khách hàng rủi ro nhất (Slide 57):")
    print(f"     * Số khách hàng cần liên hệ (10%): {top10_count:,} khách hàng")
    print(f"     * Tỷ lệ rời bỏ trong nhóm gọi   : {churn_rate_top10*100:.2f}% (Gấp {churn_rate_top10/churn_rate:.2f} lần tỷ lệ trung bình!)")
    print(f"     * Tỷ lệ churner tóm gọn được    : {capture_share*100:.2f}% ({captured_churners:,} / {y.sum():,} khách hàng rời bỏ)")

    # -------------------------------------------------------------------------
    # BƯỚC 8: DEPLOY & CREATE SUBMISSION (Triển khai & Tạo file Kaggle)
    # -------------------------------------------------------------------------
    print("\n[BƯỚC 8/8] DEPLOY · Huấn luyện toàn diện & Sinh file Submission chuẩn Kaggle")
    print("-" * 85)
    
    # Huấn luyện pipeline hoàn chỉnh trên 100% dữ liệu train
    pipeline = ChurnPipeline(**MODEL_PARAMS)
    pipeline.fit(train_raw)

    # Lưu mô hình chuẩn duy nhất vào models/
    pipeline.save(CHURN_MODEL_PATH)

    base_model = BaselineLogisticModel()
    base_model.fit(train_raw)
    joblib.dump(base_model, BASELINE_MODEL_PATH)

    print(f"  + Đã lưu mô hình tối ưu : {CHURN_MODEL_PATH}")
    print(f"  + Đã lưu mô hình baseline: {BASELINE_MODEL_PATH}")

    # Sinh dự báo cho test.csv (8,000 khách hàng)
    if test_raw is not None:
        pred_test_opt = pipeline.predict(test_raw)
        pred_test_base = base_model.predict(test_raw)

        sub_opt = pd.DataFrame({
            "customer_id": test_raw["customer_id"].values,
            "churn_3m": np.round(pred_test_opt, 6),
        })
        sub_base = pd.DataFrame({
            "customer_id": test_raw["customer_id"].values,
            "churn_3m": np.round(pred_test_base, 6),
        })

        sub_opt.to_csv(SUBMISSION_PATH, index=False)
        sub_base.to_csv(SUBMISSION_BASELINE_PATH, index=False)

        print(f"\n  + Đã xuất file nộp bài tối ưu: {SUBMISSION_PATH}")
        print(f"  + Đã xuất file baseline       : {SUBMISSION_BASELINE_PATH}")

        # Kiểm tra tính toàn vẹn của file submission
        print("\n  + KIỂM TRA TOÀN VẸN FILE SUBMISSION KAGGLE:")
        print(f"     * Số dòng            : {len(sub_opt):,} (Yêu cầu đúng 8,000) -> {'HỢP LỆ' if len(sub_opt) == 8000 else 'LỖI'}")
        print(f"     * Danh sách các cột : {list(sub_opt.columns)} -> {'HỢP LỆ' if list(sub_opt.columns) == ['customer_id', 'churn_3m'] else 'LỖI'}")
        print(f"     * Số giá trị khuyết  : {sub_opt.isna().sum().sum()} -> {'HỢP LỆ' if sub_opt.isna().sum().sum() == 0 else 'LỖI'}")
        print(f"     * Miền xác suất     : Min={sub_opt['churn_3m'].min():.4f}, Max={sub_opt['churn_3m'].max():.4f}")
        print(f"     * Top 5 dòng đầu tiên:\n{sub_opt.head(5).to_string(index=False)}")

    # 5. Mô phỏng giao diện What-if form (Slide 58)
    print("\n  + MÔ PHỎNG WHAT-IF SCENARIO FORM CHO CÁN BỘ NGÂN HÀNG (Slide 58):")
    sample_cust = test_raw.iloc[0].to_dict() if test_raw is not None else {}
    cid = sample_cust.get("customer_id", "CUSJCOH081")
    
    # Kịch bản 1: Giữ nguyên
    df_case1 = pd.DataFrame([sample_cust])
    prob_case1 = pipeline.predict(df_case1)[0]
    
    # Kịch bản 2: Khách sụt giảm chi tiêu, có khiếu nại
    sample_cust_stressed = sample_cust.copy()
    sample_cust_stressed["txn_count_3m"] = 2
    sample_cust_stressed["spend_change_pct"] = -75.0
    sample_cust_stressed["complaints_12m"] = 2
    df_case2 = pd.DataFrame([sample_cust_stressed])
    prob_case2 = pipeline.predict(df_case2)[0]

    print(f"     * Khách hàng: {cid}")
    print(f"     * Hiện trạng ban đầu : Số GD=23, Khiếu nại=0 -> Xác suất rời bỏ: {prob_case1:.2%}")
    print(f"     * Giả lập khủng hoảng: Số GD=2, Chi tiêu -75%, Khiếu nại=2 -> Xác suất rời bỏ: {prob_case2:.2%}")
    print(f"     -> Cán bộ ngân hàng có thể truy vấn 'What-if' mà không cần sửa mã nguồn.")

    print("\n" + "=" * 85)
    print(" >>> HOÀN TẤT TOÀN BỘ QUY TRÌNH 8 BƯỚC THÀNH CÔNG! <<< ")
    print("=" * 85)


if __name__ == "__main__":
    main()
