# Dự Án Dự Báo Rủi Ro Rời Bỏ Thẻ Tín Dụng (`Card Churn Prediction`)

Giải pháp **AI & Data Science chuẩn Production** dự báo xác suất khách hàng đóng thẻ tín dụng trong 3 tháng tới (`churn_3m`), đánh giá bằng **ROC-AUC** trên 5-Fold Stratified Cross-Validation.

---

## 1. Cấu Trúc Thư Mục Tối Ưu

```text
churn/
├── data/                                # Dữ liệu thô gốc Kaggle (bất biến)
│   ├── train.csv                        # 12.120 dòng (gồm 120 dòng trùng lặp & nhiều lỗi nhập liệu)
│   ├── test.csv                         # 8.000 khách hàng cần dự báo (23 cột)
│   └── sample_submission.csv            # Khung nộp bài mẫu 8.000 dòng
│
├── models/                              # Mô hình đã huấn luyện tốt nhất + đối chứng
│   ├── churn_model.pkl                  # ChurnPipeline tối ưu (Cleaner + Hybrid Multi-seed HGB & Spline GAM)
│   └── baseline_model.pkl               # BaselineLogisticModel đối chứng
│
├── notebooks/                           # Notebook nghiên cứu & giải pháp
│   ├── solution_churn.ipynb             # Notebook giải pháp hoàn chỉnh (EDA + Cleaning + ChurnPipeline)
│   └── starter_churn.ipynb              # Notebook baseline khởi điểm của cuộc thi
│
├── reports/                             # Báo cáo phân tích kiểm toán dữ liệu
│   └── phan_tich_du_lieu_churn.md
│
├── src/                                 # Mã nguồn mô-đun hóa Production
│   ├── __init__.py
│   ├── config.py                        # Quản lý đường dẫn & siêu tham số MODEL_PARAMS
│   ├── cleaning.py                      # ChurnDataCleaner (làm sạch & suy luận chéo không làm mất gốc)
│   ├── features.py                      # Tạo 58 đặc trưng hành vi & nghiệp vụ ngân hàng
│   ├── baseline_model.py                # BaselineLogisticModel
│   └── churn_model.py                   # ChurnPredictor & ChurnPipeline
│
├── submissions/                         # Kết quả dự báo Kaggle sạch (không trùng lặp)
│   ├── submission.csv                   # File nộp bài tối ưu (8.000 dòng từ test.csv)
│   └── submission_baseline.csv          # File Baseline đối chứng
│
├── venv/                                # Môi trường ảo Python riêng cho Churn
├── train.py                             # Kiểm định 5-Fold CV & huấn luyện ChurnPipeline + Baseline
├── predict.py                           # Nạp model, dự báo trên test.csv & xuất submissions/submission.csv
├── inspect_model.py                     # Kiểm tra tham số suy luận & Odds Ratio bên trong churn_model.pkl
├── requirements.txt                     # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 2. Kết Quả Kiểm Định 5-Fold Stratified CV & Kiểm Toán Dữ Liệu

Chạy kiểm chứng thực tế bằng `python train.py`:

- **Kiểm toán & làm sạch `train.csv`**: Phát hiện `788` lỗi bất thường (`unit_error_vnd_x1e6`: `351`, `negative_txn_amount`: `169`, `duplicate_row`: `120`, `invalid_sentinel_utilization`: `79`, `invalid_sentinel_age`: `69`); khử trùng lặp tập Train từ `12.120` còn `12.000` dòng và tạo `58` đặc trưng nghiệp vụ (`anniversary_unwaived`, `is_cashback_promo`, `util_is_zero`, `spend_drop`, `avg_txn_amt`...).
- **So sánh 5-Fold Stratified CV ROC-AUC**:
  - **Starter Baseline**: `0.7302`
  - **Spline GAM**: `0.8096`
  - **Multi-seed HistGradientBoosting**: `0.8093`
  - **Hybrid Ensemble (`ChurnPipeline`: 50% HGB + 50% Spline GAM)**: **`0.8129`**
- **Hiệu quả vận hành (Top 10% Call List)**: Khi gọi cho `1.200` khách hàng rủi ro cao nhất, tỷ lệ rời bỏ thực tế đạt **`51.67%`** (gấp **`4.43x`** mức trung bình `11.66%`), tóm gọn **`44.13%`** (`620 / 1.405`) tổng số khách hàng rời bỏ.
- **Chất lượng `submissions/submission.csv`**: Đúng `8.000` dòng (`['customer_id', 'churn_3m']`), `0` NaN, miền xác suất `[0.0085, 0.9449]` dự báo trực tiếp từ `test.csv`.

---

## 3. Hướng Dẫn Chạy Pipeline

> **Lưu ý về phiên bản thư viện:** File `models/churn_model.pkl` hiện tại được đóng gói bằng `scikit-learn 1.8.0` / `pandas 2.0.3`. Nếu chạy trong `venv/` (cài `scikit-learn` / `pandas` mới hơn), hãy chạy lại `python train.py` trước khi `predict.py` hoặc `inspect_model.py`.

```bash
cd data/churn

# 1. Kiểm định 5-Fold CV & huấn luyện lại churn_model.pkl + baseline_model.pkl
python train.py

# 2. Dự báo trên test.csv & xuất submissions/submission.csv
python predict.py

# 3. Kiểm tra tham số suy luận chéo & hệ số Odds Ratio bên trong churn_model.pkl
python inspect_model.py
```
