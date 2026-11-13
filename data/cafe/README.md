# Dự Án Dự Báo Doanh Thu Chuỗi Cửa Hàng Cafe (`CafeRevenueModelV3`)

Giải pháp **AI & Data Science chuẩn Production** dự báo doanh thu hàng ngày (tính bằng **triệu VNĐ**) cho 45 cửa hàng cà phê trong **Quý 1/2026** (86–90 ngày/cửa hàng sau khi loại ngày nghỉ Tết, tổng cộng **3.869 dòng**), sử dụng thước đo đánh giá **RMSLE** (*Root Mean Squared Logarithmic Error*).

---

## 1. Cấu Trúc Thư Mục Tối Ưu

```text
cafe/
├── data/                                    # Dữ liệu thô gốc Kaggle (bất biến)
│   ├── train.csv                            # 26.632 dòng lịch sử 2024–2025 (chứa nhiễu DGP chủ đích)
│   ├── test.csv                             # 3.869 dòng Q1/2026 với is_promo thực tế (380 ngày, 9.8%)
│   ├── stores.csv                           # 45 cửa hàng (diện tích, ghế, thành phố, loại hình)
│   ├── weather.csv                          # 1.593 quan sát thời tiết hàng ngày theo thành phố
│   ├── holidays.csv                         # 55 sự kiện/ngày lễ, chu kỳ Tết
│   └── sample_submission.csv                # Khung nộp bài mẫu 3.869 dòng
│
├── docs/                                    # Tài liệu chuẩn phương pháp luận
│   └── Chapter 4 - Building models and developing AI applications.pdf
│
├── models/                                  # Mô hình đã huấn luyện tốt nhất + đối chứng
│   ├── cafe_revenue_model_v3.pkl            # Mô hình AI tối ưu thế hệ 3 (CafeRevenueModelV3)
│   └── baseline_model.pkl                   # Mô hình Baseline trung vị (BaselineWeekdayModel)
│
├── notebooks/                               # 2 bộ notebook chính
│   ├── 01_bao_cao_chuyen_sau/               # Chuỗi 5 phần báo cáo chuyên sâu P1 → P5
│   │   ├── P1_Cleandata_compare.ipynb
│   │   ├── P2_So_sanh_2024_2025_2026.ipynb
│   │   ├── P3_Feature_Engineering_Tet_Cycles.ipynb
│   │   ├── P4_Model_Development_Validation.ipynb
│   │   └── P5_Forecast_2026_Executive_Report.ipynb
│   └── 02_quy_trinh_8_buoc_chapter4/        # Quy trình 8 bước chuẩn AI lifecycle (Step 1 → Step 8)
│       ├── Step1_Collect_Prepare.ipynb
│       ├── Step2_Explore_EDA.ipynb
│       ├── Step3_Preprocess_Clean.ipynb
│       ├── Step4_Split_Validation.ipynb
│       ├── Step5_Scale_Feature_Engineering.ipynb
│       ├── Step6_Model_Train.ipynb
│       ├── Step7_Evaluate_Diagnostics.ipynb
│       └── Step8_Deploy_Submission.ipynb
│
├── reports/                                 # Báo cáo tổng hợp & 8 biểu đồ chuẩn xuất bản
│   ├── bao_cao_tong_quan_du_an_cafe.md
│   ├── phan_tich_du_lieu_cafe.md
│   ├── Bao_Cao_Tong_Quan_Du_An_Cafe_Revenue.docx
│   └── figures/                             # fig1_cleaning_comparison.png .. fig8_heatmap_city_type.png
│
├── src/                                     # Mã nguồn mô-đun hóa Production
│   ├── __init__.py
│   ├── config.py                            # Quản lý đường dẫn & EVAL_QUARTERS
│   ├── cleaning.py                          # Làm sạch 5 bảng (clean_train, clean_test, build_test_grid_if_missing...)
│   ├── features.py                          # Đặc trưng phân tích dùng cho P3 & Step5
│   ├── baseline_model.py                    # BaselineWeekdayModel
│   └── cafe_model_v3.py                     # CafeRevenueModelV3 (Hierarchical Ridge + AR(1) + HistGBDT + Taylor)
│
├── submissions/                             # Kết quả dự báo Kaggle sạch (không trùng lặp)
│   ├── submission_v3.csv                    # File nộp bài tối ưu Q1/2026 (3.869 dòng)
│   └── submission_baseline.csv              # File Baseline dùng để đối chứng
│
├── venv/                                    # Môi trường ảo Python riêng cho Cafe
├── train.py                                 # Kiểm định 4 quý 2025 & huấn luyện CafeRevenueModelV3 + Baseline
├── predict.py                               # Tải model, kiểm toán test.csv & xuất submissions/submission_v3.csv
├── inspect_model.py                         # Kiểm tra tham số nội tại của baseline & cafe_revenue_model_v3.pkl
├── requirements.txt                         # Danh sách thư viện phụ thuộc
└── README.md
```

---

## 2. Kết Quả Kiểm Định Chéo Out-of-Time (4 Quý Năm 2025)

Chạy kiểm chứng thực tế bằng `python train.py`:

| Giai đoạn kiểm định | Baseline (`BaselineWeekdayModel`) | AI Tối ưu (`CafeRevenueModelV3`) | Mức cải thiện |
| :--- | :---: | :---: | :---: |
| **Q1/2025** | `0.2758` | `0.1045` | **-62.12%** |
| **Q2/2025** | `0.2391` | `0.1027` | **-57.04%** |
| **Q3/2025** | `0.2750` | `0.1077` | **-60.84%** |
| **Q4/2025** | `0.2275` | `0.1123` | **-50.64%** |
| **Trung bình 4 quý** | **`0.2544`** | **`0.1068`** | **-57.66%** |

- **Làm sạch tự động `train.csv`**: Phát hiện và sửa `132` dòng trùng lặp, `30` lỗi `1000x`, `53` lỗi `10x`, và `26` lỗi `0.1x` hoàn toàn trên bản sao trong bộ nhớ, giữ nguyên `data/train.csv` gốc.
- **Chất lượng `submissions/submission_v3.csv`**: Đúng `3.869` dòng (`['id', 'revenue']`), `0` NaN, `0` giá trị âm, Min = `1.92` tr, Median = `19.81` tr, Max = `69.90` tr.

---

## 3. Hướng Dẫn Chạy Pipeline

> **Lưu ý về phiên bản thư viện:** File `models/cafe_revenue_model_v3.pkl` hiện tại được đóng gói bằng `scikit-learn 1.8.0` / `pandas 2.0.3`. Nếu chạy trong `venv/` (cài `scikit-learn` / `pandas` mới hơn), hãy chạy lại `python train.py` trước khi `predict.py` hoặc `inspect_model.py` để đồng bộ định dạng pickle theo đúng môi trường.

```bash
cd data/cafe

# 1. Huấn luyện & kiểm định chéo 4 quý (tạo lại cafe_revenue_model_v3.pkl & baseline_model.pkl)
python train.py

# 2. Sinh file nộp bài chuẩn Kaggle (submissions/submission_v3.csv)
python predict.py

# 3. Kiểm tra tham số bên trong mô hình đã lưu
python inspect_model.py
```
