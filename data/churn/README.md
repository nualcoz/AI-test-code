# Dự Án Dự Báo Rời Bỏ Thẻ Tín Dụng (Card Churn Prediction — Kaggle Competition)

Dự án AI phân tích và dự báo xác suất khách hàng đóng thẻ tín dụng (`churn_3m = 1`) trong vòng 3 tháng kể từ ngày chốt dữ liệu (30/06/2026) theo thang đo **ROC-AUC**.

---

## 1. Cấu Trúc Thư Mục Phân Chia (Project Layout)

Thư mục đã được tổ chức theo chuẩn Machine Learning chuyên nghiệp:

```text
E:\study\AI\test 1\Kaggle_Competition\data\churn/
│
├── data/                          # Toàn bộ dữ liệu đề bài
│   ├── train.csv                  # Dữ liệu khách hàng lịch sử (12.120 dòng, 24 cột)
│   ├── sample_submission.csv      # Mẫu nộp bài Kaggle (8.000 dòng: customer_id, churn_3m)
│   └── test.csv                   # (Tùy chọn) Tập kiểm thử khi chạy trên Kaggle
│
├── src/                           # Mã nguồn Python dạng module tái sử dụng
│   ├── __init__.py                # Package export
│   ├── config.py                  # Đường dẫn, cấu hình siêu tham số và phân chia CV
│   ├── cleaning.py                # Pipeline làm sạch dữ liệu (sửa lỗi x1e6, lọc sentinel, điền khuyết chéo)
│   ├── features.py                # Kỹ thuật đặc trưng nghiệp vụ (vách đá phí thường niên, săn hoàn tiền, log)
│   ├── baseline_model.py          # Mô hình Baseline Logistic Regression chuẩn starter_churn.ipynb
│   └── churn_model.py             # Mô hình AI Hybrid Ensemble (Multi-seed HGB + Spline GAM)
│
├── notebooks/                     # Toàn bộ Jupyter Notebook phân tích & thực nghiệm
│   ├── starter_churn.ipynb        # Notebook gốc của cuộc thi
│   ├── phan_tich_du_lieu_churn.ipynb # Notebook EDA kiểm toán lỗi và phân tích chuyên sâu
│   └── solution_churn.ipynb       # Notebook giải pháp tối ưu kèm kiểm thử 5-fold CV
│
├── models/                        # Trọng số mô hình đã huấn luyện (.pkl)
│   ├── baseline_model.pkl         # Mô hình Baseline lưu trữ
│   └── churn_model.pkl            # Mô hình ChurnPipeline đã tối ưu
│
├── submissions/                   # Kết quả dự báo nộp lên Kaggle
│   ├── submission.csv             # File nộp bài tối ưu (ROC-AUC ~ 0.8129)
│   └── submission_baseline.csv    # File nộp bài baseline gốc (ROC-AUC ~ 0.7302)
│
├── reports/                       # Báo cáo phân tích chuyên sâu
│   └── phan_tich_du_lieu_churn.md # Báo cáo toàn diện phân tích dữ liệu và mô hình hóa
│
├── train.py                       # Script chạy huấn luyện mô hình & kiểm định chéo 5-Fold
├── predict.py                     # Script sinh file dự báo submission.csv
├── inspect_model.py               # Script xem chi tiết nội dung các file models (.pkl)
├── submission.csv                 # File nộp bài Kaggle chính thức tại thư mục gốc
└── README.md                      # Hướng dẫn chi tiết dự án
```

---

## 2. Hướng Dẫn Sử Dụng (Quick Start)

### Huấn Luyện & Kiểm Định Chéo 5-Fold (Train)
Chạy lệnh sau trong terminal:
```bash
python train.py
```
*Script sẽ kiểm toán dữ liệu, chạy kiểm định chéo 5-Fold Stratified CV, in bảng so sánh chi tiết và lưu mô hình vào thư mục `models/`.*

### Sinh File Dự Báo Nộp Bài (Predict)
Chạy lệnh sau trong terminal:
```bash
python predict.py
```
*Script sẽ nạp mô hình từ `models/`, tính toán dự báo cho toàn bộ 8.000 khách hàng và lưu kết quả vào `submissions/submission.csv` (đồng thời cập nhật `submission.csv` ở thư mục gốc).*

### Xem Nội Dung Bên Trong Các File Models (Inspect)
Chạy lệnh sau trong terminal:
```bash
python inspect_model.py
```
*Script sẽ đọc các file `models/*.pkl` và hiển thị chi tiết các tỷ số nghiệp vụ, tỷ số nguy cơ (Odds Ratio) của các yếu tố rủi ro.*

---

## 3. Hiệu Năng So Với Baseline (5-Fold Stratified CV)

| Mô Hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF ROC-AUC | Cải Thiện |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Gốc (starter_churn)** | 0.7045 | 0.7418 | 0.7314 | 0.7378 | 0.7344 | **0.7302** | — |
| **Logistic Sau Làm Sạch** | 0.7138 | 0.7496 | 0.7533 | 0.7364 | 0.7418 | **0.7389** | +0.87% |
| **Spline GAM + Tương Tác** | 0.7820 | 0.8282 | 0.8187 | 0.8163 | 0.8053 | **0.8096** | +7.94% |
| **Multi-Seed HGB (5 seeds)** | 0.7831 | 0.8255 | 0.8139 | 0.8163 | 0.8098 | **0.8093** | +7.91% |
| **Hybrid Ensemble (HGB + GAM)** | **0.7860** | **0.8304** | **0.8195** | **0.8192** | **0.8117** | **0.8129** | **+8.27%** |

---

## 4. Các Phát Hiện & Kỹ Thuật Đột Phá

1. **Khắc phục lỗi đơn vị $\times 10^6$**: Phát hiện 351 khách hàng có thu nhập và hạn mức ghi nhầm đơn vị VNĐ thay vì triệu VNĐ.
2. **Khắc phục giá trị Sentinel**: Lọc các giá trị tuổi bất thường (`0, 1, 150, 200, 999`), tỷ lệ sử dụng âm/vượt trần (`-1.0, 15.0`) và tiền giao dịch âm do hoàn tiền.
3. **Hiệu ứng Vách đá phí thường niên (`anniversary_unwaived`)**: Khách hàng mở thẻ vào các tháng 7, 8, 9 và không được miễn phí thường niên có tỷ lệ đóng thẻ vọt lên **25.1%** (Odds ratio x1.61).
4. **Hiệu ứng Khách săn hoàn tiền (`is_cashback_promo`)**: Khách dùng thẻ Cashback vừa hết 60 ngày ưu đãi có tỷ lệ rời bỏ lên **24.2%** (Odds ratio x1.61).
5. **Tổ hợp Hybrid Ensemble**: Kết hợp cây quyết định phân đoạn cục bộ (HistGradientBoosting) với đường cong trơn toàn cục (Spline GAM) giúp mô hình đạt ROC-AUC **0.8129**, không bị quá khớp.
