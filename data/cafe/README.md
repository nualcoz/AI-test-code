# Dự Án Dự Báo Doanh Thu Chuỗi Cửa Hàng Cafe (Kaggle Competition)

Dự án AI phân tích và dự báo doanh thu hàng ngày cho từng cửa hàng trong chuỗi cà phê cho Quý 1/2026 (01/01/2026 đến 31/03/2026) theo thang đo **RMSLE** trên $\log(1 + \text{revenue})$.

---

## 1. Cấu Trúc Dự Án Đã Được Dọn Dẹp & Phân Nhóm Chuẩn Mực

Toàn bộ thư mục đã được làm sạch, loại bỏ hoàn toàn các file và thư mục trùng lặp, chia thành từng nhóm chức năng rõ ràng:

```text
cafe/
│
├── data/                               # Dữ liệu cuộc thi chính thức
│   ├── train.csv                       # Lịch sử doanh thu 2024-2025 (26.632 dòng)
│   ├── test.csv                        # Tập kiểm thử Quý 1/2026 có nhãn promo (3.869 dòng)
│   ├── stores.csv                      # Hồ sơ 45 cửa hàng (diện tích, chỗ ngồi, loại quán)
│   ├── weather.csv                     # Chuỗi thời tiết Hà Nội, TP.HCM, Đà Nẵng
│   ├── holidays.csv                    # Danh mục ngày lễ, sự kiện và Tết Âm lịch
│   └── sample_submission.csv           # Mẫu nộp bài Kaggle (3.869 dòng)
│
├── notebooks/                          # Toàn bộ Jupyter Notebooks (Phân chia 3 nhóm khoa học)
│   │
│   ├── 01_bao_cao_chuyen_sau/          # [NHÓM 1] Chuỗi 5 Notebooks xây dựng Báo cáo Tổng quan (P1 -> P5)
│   │   ├── P1_Cleandata_compare.ipynb  # Phần 1: Làm sạch dữ liệu, công cụ & so sánh Before-After
│   │   ├── P2_So_sanh_2024_2025_2026.ipynb # Phần 2: So sánh toàn diện lịch sử 2024, 2025 với 2026
│   │   ├── P3_Feature_Engineering_Tet_Cycles.ipynb # Phần 3: Kỹ thuật đặc trưng chu kỳ Tết & thời tiết
│   │   ├── P4_Model_Development_Validation.ipynb # Phần 4: Huấn luyện, tối ưu mô hình & kiểm định 4 Quý
│   │   └── P5_Forecast_2026_Executive_Report.ipynb # Phần 5: Dự báo Q1/2026 & Báo cáo tổng quan kinh doanh
│   │
│   ├── 02_quy_trinh_8_buoc_chapter4/   # [NHÓM 2] Chuỗi 8 Notebooks chuẩn giáo trình Chapter 4 (NEU FIT)
│   │   ├── Step1_Collect_Prepare.ipynb # Step 1: Đọc dữ liệu, merge bảng & Data Dictionary
│   │   ├── Step2_Explore_EDA.ipynb     # Step 2: Khám phá cấu trúc, missing, biểu đồ phân phối
│   │   ├── Step3_Preprocess_Clean.ipynb# Step 3: Tiền xử lý, khử lỗi 1000x, 10x, 0.1x, độ F, one-hot
│   │   ├── Step4_Split_Validation.ipynb# Step 4: Time-based split 4 quý năm 2025 (Rule 1)
│   │   ├── Step5_Scale_Feature_Engineering.ipynb # Step 5: Log1p, StandardScaler, tương tác đặc trưng (Rule 2)
│   │   ├── Step6_Model_Train.ipynb     # Step 6: Model zoo (Baseline, Ridge, GBDT, CafeModelV3) (Rule 3)
│   │   ├── Step7_Evaluate_Diagnostics.ipynb # Step 7: Đánh giá RMSLE, Actual vs Pred, Residuals (Rule 4)
│   │   └── Step8_Deploy_Submission.ipynb # Step 8: Lưu mô hình joblib, Colab Form What-If, xuất file
│   │
│   └── 03_notebooks_tham_khao/         # [NHÓM 3] Notebooks tổng hợp & nghiên cứu ban đầu
│       ├── solution_with_test.ipynb    # Bản giải pháp tổng hợp tích hợp test.csv
│       ├── starter_cafe.ipynb          # Notebook starter gốc từ Kaggle
│       ├── phan_tich_du_lieu_cafe.ipynb# Phân tích EDA chuyên sâu ban đầu
│       ├── solution_cafe.ipynb         # Giải pháp V1 (trước khi có test.csv)
│       └── solution_part_2.ipynb       # Giải pháp V2 (trước khi có test.csv)
│
├── reports/                            # Báo cáo phân tích chuyên sâu & quản trị kinh doanh
│   ├── bao_cao_tong_quan_du_an_cafe.md # [MỚI] Báo cáo tổng quan hoàn chỉnh dự án Cafe Revenue
│   └── phan_tich_du_lieu_cafe.md       # Báo cáo phân tích EDA ban đầu
│
├── docs/                               # Tài liệu học tập & slide bài giảng
│   └── Chapter 4 - Building models and developing AI applications.pdf # Giáo trình Chapter 4 NEU FIT
│
├── scripts/                            # Các công cụ script tự động sinh và thực thi notebook
│   ├── build_full_report_pipeline.py   # Script sinh chuỗi P1 -> P5
│   ├── build_chapter4_steps.py         # Script sinh chuỗi Step 1 -> Step 8
│   ├── build_p1_cleaning_notebook.py   # Script sinh P1
│   └── generate_and_run_notebook.py    # Script sinh solution_with_test.ipynb
│
├── src/                                # Mã nguồn module Python tái sử dụng
│   ├── __init__.py                     # Package export
│   ├── config.py                       # Đường dẫn, cấu hình siêu tham số và phân chia kiểm thử
│   ├── cleaning.py                     # Pipeline làm sạch dữ liệu (clean_train, clean_stores, clean_test)
│   ├── features.py                     # Xây dựng ma trận đặc trưng
│   ├── baseline_model.py               # Mô hình Baseline Day-of-Week Median
│   ├── cafe_model.py                   # Mô hình CafeRevenueModel V1
│   ├── cafe_model_v2.py                # Mô hình CafeRevenueModel V2
│   └── cafe_model_v3.py                # Mô hình CafeRevenueModel V3 tối ưu
│
├── models/                             # Trọng số mô hình đã huấn luyện (.pkl)
│   ├── cafe_revenue_model_v3.pkl       # [TỐI ƯU] Trọng số CafeRevenueModelV3 (RMSLE 0.1068)
│   ├── cafe_revenue_model_v2.pkl
│   ├── cafe_revenue_model.pkl
│   └── baseline_model.pkl
│
├── submissions/                        # Kết quả dự báo nộp lên Kaggle
│   ├── submission_v3.csv               # File nộp bài tối ưu mới nhất
│   ├── submission.csv                  # File nộp bài chính thức
│   └── submission_baseline.csv         # File nộp bài theo baseline
│
├── train.py                            # Script chạy huấn luyện mô hình & kiểm định chéo 4 quý
├── predict.py                          # Script sinh file dự báo submission.csv trực tiếp từ test.csv
├── inspect_model.py                    # Script xem chi tiết nội dung các file models (.pkl)
├── submission.csv                      # File nộp bài Kaggle chính thức (ở thư mục gốc)
└── README.md                           # Hướng dẫn chi tiết dự án
```

---

## 2. Hướng Dẫn Sử Dụng Nhanh

### Chạy Dự Báo Nộp Bài (Kaggle Submission):
```bash
python predict.py
```
*Script nạp `data/test.csv`, áp dụng mô hình tối ưu `models/cafe_revenue_model_v3.pkl`, kiểm định toàn vẹn và xuất ra `submission.csv` (3.869 dòng).*

### Huấn Luyện & Kiểm Định Chéo 4 Quý (Backtesting):
```bash
python train.py
```
*Kiểm định Out-of-Time trên 4 Quý năm 2025, in bảng so sánh RMSLE và lưu mô hình vào `models/`.*

### Đọc Báo Cáo Tổng Quan Dự Án:
Mở file markdown hoàn chỉnh tại:
👉 `reports/bao_cao_tong_quan_du_an_cafe.md`
