# Dự Án Dự Báo Doanh Thu Chuỗi Cửa Hàng Cafe (Kaggle Competition)

Dự án AI phân tích và dự báo doanh thu hàng ngày cho từng cửa hàng trong chuỗi cà phê cho Quý 1/2026 (01/01/2026 đến 31/03/2026) theo thang đo **RMSLE** trên $\log(1 + \text{revenue})$.

---

## 1. Cấu Trúc Thư Mục Phân Chia (Project Layout)

Thư mục đã được tổ chức theo chuẩn Machine Learning chuyên nghiệp:

```text
data/cafe/
│
├── data/                          # Toàn bộ dữ liệu đề bài
│   ├── train.csv                  # Dữ liệu lịch sử 2024-2025 (date, store_id, is_promo, revenue)
│   ├── test.csv                   # Dữ liệu kiểm thử Quý 1/2026 (id, date, store_id, is_promo)
│   ├── stores.csv                 # Thông tin 45 cửa hàng (loại quán, vị trí, diện tích, chỗ ngồi)
│   ├── weather.csv                # Thời tiết hàng ngày tại Hà Nội, TP.HCM, Đà Nẵng
│   ├── holidays.csv               # Danh sách ngày lễ, Tết Âm lịch và sự kiện
│   └── sample_submission.csv      # Mẫu nộp bài (3.869 dòng: id, revenue)
│
├── src/                           # Mã nguồn Python dạng module tái sử dụng
│   ├── __init__.py                # Package export
│   ├── config.py                  # Đường dẫn, cấu hình siêu tham số và phân chia kiểm thử
│   ├── cleaning.py                # Pipeline xử lý làm sạch dữ liệu (F->C, lỗi 10x, 1000x, clean_test)
│   ├── features.py                # Xây dựng ma trận đặc trưng (thời tiết, chu kỳ Tết, khai trương, tương tác)
│   ├── baseline_model.py          # Mô hình Baseline Day-of-Week Median chuẩn starter_cafe.ipynb
│   ├── cafe_model.py              # Mô hình AI CafeRevenueModel (V1)
│   ├── cafe_model_v2.py           # Mô hình AI CafeRevenueModel (V2)
│   └── cafe_model_v3.py           # Mô hình AI CafeRevenueModel (V3 tối ưu khi có test.csv)
│
├── notebooks_pipeline/            # [MỚI] Chuỗi 8 Notebooks chuẩn giáo trình Chapter 4 (NEU FIT)
│   ├── P1_Cleandata_compare.ipynb # Chuyên đề: Làm sạch dữ liệu, công cụ & so sánh Before-After
│   ├── Step1_Collect_Prepare.ipynb# Step 1: Collect & prepare (bài toán, đọc bảng, merge, dictionary)
│   ├── Step2_Explore_EDA.ipynb    # Step 2: Explore (info, describe, missing, charts, correlation)
│   ├── Step3_Preprocess_Clean.ipynb# Step 3: Preprocess (khử 1000x, 10x, 0.1x, sửa độ F, get_dummies)
│   ├── Step4_Split_Validation.ipynb# Step 4: Split (Time-based split 4 quý 2025, Rule 1)
│   ├── Step5_Scale_Feature_Engineering.ipynb # Step 5: Scale & Features (log1p, StandardScaler, Rule 2)
│   ├── Step6_Model_Train.ipynb    # Step 6: Model & train (Baseline, Ridge, GBDT, V3, Rule 3)
│   ├── Step7_Evaluate_Diagnostics.ipynb # Step 7: Evaluate (RMSLE 4 quý, Actual vs Pred, Rule 4)
│   └── Step8_Deploy_Submission.ipynb # Step 8: Deploy (joblib, What-If Colab form, submission)
│
├── notebooks/                     # Toàn bộ Jupyter Notebook phân tích & thực nghiệm
│   ├── steps/                     # Bản sao đồng bộ 8 bước chuẩn Chapter 4
│   ├── P1_Cleandata_compare.ipynb # Bản sao đồng bộ Phần 1 để truy cập nhanh
│   ├── solution_with_test.ipynb   # Notebook tính toán & dự báo sau khi dùng file test thực tế
│   ├── solution_part_2.ipynb      # Notebook phân tích kiến trúc giải pháp nâng cao
│   ├── solution_cafe.ipynb        # Notebook giải pháp tối ưu kèm kiểm thử 4 quý
│   ├── phan_tich_du_lieu_cafe.ipynb # Notebook EDA phân tích chuyên sâu 6 đặc trưng
│   └── starter_cafe.ipynb         # Notebook gốc của đề bài
│
├── models/                        # Trọng số mô hình đã huấn luyện
│   ├── baseline_model.pkl         # Mô hình Baseline lưu trữ
│   ├── cafe_revenue_model.pkl     # Mô hình CafeRevenueModel V1
│   ├── cafe_revenue_model_v2.pkl  # Mô hình CafeRevenueModel V2
│   └── cafe_revenue_model_v3.pkl  # [MỚI] Mô hình CafeRevenueModel V3 tối ưu
│
├── submissions/                   # Kết quả dự báo nộp lên Kaggle
│   ├── submission_v3.csv          # [MỚI] File nộp bài tối ưu sau khi tích hợp test.csv (RMSLE ~ 0.106)
│   ├── submission_v2.csv          # File nộp bài theo mô hình V2
│   ├── submission.csv             # File nộp bài chính thức đồng bộ
│   └── submission_baseline.csv    # File nộp bài theo baseline gốc
│
├── reports/                       # Báo cáo phân tích chuyên sâu
│   └── phan_tich_du_lieu_cafe.md  # Báo cáo toàn diện phân tích dữ liệu và mô hình hóa
│
├── train.py                       # Script chạy huấn luyện mô hình & kiểm định chéo 4 quý
├── predict.py                     # Script sinh file dự báo submission.csv trực tiếp từ test.csv
├── generate_and_run_notebook.py   # Script tự động tạo và thực thi notebook solution_with_test.ipynb
├── submission.csv                 # File nộp bài Kaggle chính thức (3.869 dòng)
└── README.md                      # Hướng dẫn chi tiết dự án
```

---

## 2. Hướng Dẫn Sử Dụng (Quick Start)

### Huấn Luyện & Kiểm Định Chéo 4 Quý (Train)
Chạy lệnh sau trong terminal:
```bash
python train.py
```
*Script sẽ kiểm định mô hình qua 4 quý năm 2025 (Q1, Q2, Q3, Q4), in bảng so sánh chi tiết và lưu mô hình vào `models/cafe_revenue_model_v3.pkl`.*

### Sinh File Dự Báo Nộp Bài (Predict)
Chạy lệnh sau trong terminal:
```bash
python predict.py
```
*Script sẽ nạp trực tiếp tập kiểm thử thực tế `data/test.csv`, áp dụng mô hình `CafeRevenueModelV3`, tính toán dự báo cho toàn bộ 3.869 dòng của Quý 1/2026 và lưu kết quả vào `submissions/submission_v3.csv` đồng thời cập nhật `submission.csv` ở thư mục gốc.*

### Chạy Lại Notebook Tính Toán (Solution With Test)
```bash
python generate_and_run_notebook.py
```
*Script sẽ tự động khởi tạo, thực thi và nhúng toàn bộ bảng biểu, biểu đồ và kết quả thống kê vào `notebooks/solution_with_test.ipynb`.*

---

## 3. Hiệu Năng So Với Baseline (Out-of-Time CV 4 Quý Năm 2025)

| Mô Hình | Q1/2025 | Q2/2025 | Q3/2025 | Q4/2025 | Trung Bình 4 Quý | Cải Thiện So Với Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Gốc (Chưa lọc lỗi)** | 0.2758 | 0.2391 | 0.2750 | 0.2275 | **0.2544** | — |
| **CafeRevenueModel V1** | 0.1045 | 0.1024 | 0.1066 | 0.1117 | **0.1063** | +58.2% |
| **CafeRevenueModel V3 (Có test.csv)** | **0.1045** | **0.1027** | **0.1077** | **0.1123** | **0.1068** | **+57.7%** |

---

## 4. Đặc Điểm Kỹ Thuật Nổi Bật Sau Khi Tích Hợp File Test

1. **Khai thác nhãn khuyến mãi thực tế `test.csv`**:
   - Tận dụng chính xác 380 ngày khuyến mãi (`is_promo = 1`) trong Quý 1/2026 thay vì phải giả định xác suất mềm như trước.
   - Nắm bắt chính xác đợt siêu khuyến mãi tập trung đầu năm (05/01 - 08/01/2026 với hơn 30 quán đồng loạt chạy promo).
2. **Quy trình làm sạch toàn diện**:
   - Khử nhiễu nhập liệu DGP ($1000\times, 10\times, 0.1\times$).
   - Chuẩn hóa nhiệt độ Fahrenheit sang Celsius, điền khuyết diện tích bằng tỷ lệ median $m^2/\text{ghế}$.
3. **Kiến trúc Hierarchical Multi-Stage**:
   - Stage 1: Ridge Regression với trọng số mẫu giảm thiểu ảnh hưởng của thời tiết nội suy.
   - Dual-Window Empirical Bayes Intercepts: Kết hợp $HL_{\text{long}} = 365$ ngày và $HL_{\text{mid}} = 50$ ngày, co cụm James-Stein về Bayesian Ridge Prior từ metadata quán.
   - Kalman AR(1) State Propagation: Lan truyền độ trôi ngắn hạn $\rho = 0.80$ sang các tuần đầu Q1/2026.
   - Stage 2: HistGradientBoostingRegressor học bù phần dư phi tuyến với nhãn khuyến mãi chính xác.
   - Exact Taylor Correction cho RMSLE: Tối ưu trực tiếp hàm mục tiêu $\log(1 + y)$.
