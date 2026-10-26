# Kaggle Competitions: Cafe Revenue & Card Churn Solutions

Kho lưu trữ chứa toàn bộ mã nguồn, dữ liệu, mô hình và báo cáo phân tích chuyên sâu cho 2 bài toán Kaggle:
1. **Cafe Revenue Prediction** (`data/cafe`): Bài toán Hồi quy (Regression) dự báo doanh thu chuỗi quán cafe theo thang đo **RMSLE**.
2. **Card Churn Prediction** (`data/churn`): Bài toán Phân loại (Binary Classification) dự báo xác suất khách hàng đóng thẻ tín dụng theo thang đo **ROC-AUC**.

---

## 1. Cấu Trúc Tổng Quan Repository

```text
E:\study\AI\test 1\Kaggle_Competition/
│
├── data/
│   ├── cafe/                      # Dự án 1: Cafe Revenue (Regression)
│   │   ├── data/                  # train.csv, stores.csv, weather.csv, holidays.csv
│   │   ├── src/                   # config, cleaning, features, baseline_model, cafe_model
│   │   ├── notebooks/             # starter, eda, solution notebooks
│   │   ├── models/                # baseline_model.pkl, cafe_revenue_model.pkl
│   │   ├── submissions/           # submission.csv, submission_baseline.csv
│   │   ├── reports/               # phan_tich_du_lieu_cafe.md
│   │   ├── train.py               # Huấn luyện & CV 4 quý (RMSLE ~ 0.106)
│   │   ├── predict.py             # Sinh file submission Q1/2026
│   │   └── inspect_model.py       # Khám phá trọng số mô hình
│   │
│   └── churn/                     # Dự án 2: Card Churn (Classification)
│       ├── data/                  # train.csv, sample_submission.csv
│       ├── src/                   # config, cleaning, features, baseline_model, churn_model
│       ├── notebooks/             # starter, eda, solution notebooks
│       ├── models/                # baseline_model.pkl, churn_model.pkl
│       ├── submissions/           # submission.csv, submission_baseline.csv
│       ├── reports/               # phan_tich_du_lieu_churn.md
│       ├── train.py               # Huấn luyện & 5-Fold Stratified CV (ROC-AUC ~ 0.8129)
│       ├── predict.py             # Sinh file submission
│       └── inspect_model.py       # Khám phá trọng số mô hình
│
├── venv/                          # Môi trường ảo Python (Virtual Environment)
├── requirements.txt               # Danh sách thư viện cần thiết
├── .gitignore                     # Cấu hình bỏ qua venv, cache, file tạm
└── README.md                      # Tài liệu tổng quan dự án
```

---

## 2. Hướng Dẫn Kích Hoạt Môi Trường Ảo (Virtual Environment)

Môi trường ảo `venv` đã được cài đặt sẵn tại thư mục gốc.

### Trên Windows (PowerShell):
```powershell
# Kích hoạt venv
.\venv\Scripts\Activate.ps1

# (Nếu gặp lỗi ExecutionPolicy, chạy lệnh sau một lần: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
```

### Trên Windows (Command Prompt - cmd):
```cmd
venv\Scripts\activate.bat
```

### Cài đặt lại thư viện khi cần:
```bash
pip install -r requirements.txt
```

---

## 3. Cách Chạy Từng Dự Án

### 3.1. Dự án Cafe Revenue (Regression — Thang đo RMSLE)
```powershell
cd data/cafe

# 1. Huấn luyện và kiểm định chéo 4 quý
python train.py

# 2. Sinh file nộp bài submission.csv
python predict.py

# 3. Xem các hệ số thời tiết, khuyến mãi, doanh thu quán
python inspect_model.py
```
- **Hiệu năng**: Giảm RMSLE từ **0.2363** (Baseline) xuống **0.1063** (CafeRevenueModel, cải thiện **+57.9%**).

---

### 3.2. Dự án Card Churn (Classification — Thang đo ROC-AUC)
```powershell
cd data/churn

# 1. Huấn luyện và kiểm định chéo 5-Fold Stratified CV
python train.py

# 2. Sinh file nộp bài submission.csv
python predict.py

# 3. Xem các tỷ số nguy cơ (Odds Ratio) phí thường niên, hoàn tiền
python inspect_model.py
```
- **Hiệu năng**: Nâng ROC-AUC từ **0.7302** (Baseline) lên **0.8129** (Hybrid Ensemble GBDT + Spline GAM, cải thiện **+8.27%**).
