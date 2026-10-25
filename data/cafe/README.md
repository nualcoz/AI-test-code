# Dự Án Dự Báo Doanh Thu Chuỗi Cửa Hàng Cafe (Kaggle Competition)

Dự án AI phân tích và dự báo doanh thu hàng ngày cho từng cửa hàng trong chuỗi cà phê cho Quý 1/2026 (01/01/2026 đến 31/03/2026) theo thang đo **RMSLE** trên $\log(1 + \text{revenue})$.

---

## 1. Cấu Trúc Thư Mục Phân Chia (Project Layout)

Thư mục đã được tổ chức theo chuẩn Machine Learning chuyên nghiệp:

```text
E:\study\AI\Kaggle_Competition-20261025T084058Z-1-001\Kaggle_Competition\data\cafe/
│
├── data/                          # Toàn bộ dữ liệu đề bài
│   ├── train.csv                  # Dữ liệu lịch sử 2024-2025 (date, store_id, is_promo, revenue)
│   ├── stores.csv                 # Thông tin 46 cửa hàng (loại quán, vị trí, diện tích, chỗ ngồi)
│   ├── weather.csv                # Thời tiết hàng ngày tại Hà Nội, TP.HCM, Đà Nẵng
│   ├── holidays.csv               # Danh sách ngày lễ, Tết Âm lịch và sự kiện
│   └── sample_submission.csv      # Mẫu nộp bài (3.869 dòng: id, revenue)
│
├── src/                           # Mã nguồn Python dạng module tái sử dụng
│   ├── __init__.py                # Package export
│   ├── config.py                  # Đường dẫn, cấu hình siêu tham số và phân chia kiểm thử
│   ├── cleaning.py                # Pipeline xử lý làm sạch dữ liệu (F->C, lỗi 10x, 1000x, khuyết tật)
│   ├── features.py                # Xây dựng ma trận đặc trưng (thời tiết, chu kỳ Tết, khai trương, tương tác)
│   ├── baseline_model.py          # Mô hình Baseline Day-of-Week Median chuẩn starter_cafe.ipynb
│   └── cafe_model.py              # Mô hình AI CafeRevenueModel tối ưu leaderboard
│
├── notebooks/                     # Toàn bộ Jupyter Notebook phân tích & thực nghiệm
│   ├── starter_cafe.ipynb         # Notebook gốc của đề bài
│   ├── phan_tich_du_lieu_cafe.ipynb # Notebook EDA phân tích chuyên sâu 6 đặc trưng
│   └── solution_cafe.ipynb        # Notebook giải pháp tối ưu kèm kiểm thử 4 quý
│
├── models/                        # Trọng số mô hình đã huấn luyện
│   ├── baseline_model.pkl         # Mô hình Baseline lưu trữ
│   └── cafe_revenue_model.pkl     # Mô hình CafeRevenueModel đã tối ưu
│
├── submissions/                   # Kết quả dự báo nộp lên Kaggle
│   ├── submission.csv             # File nộp bài tối ưu (RMSLE ~ 0.106)
│   └── submission_baseline.csv    # File nộp bài theo baseline gốc
│
├── reports/                       # Báo cáo phân tích chuyên sâu
│   └── phan_tich_du_lieu_cafe.md  # Báo cáo toàn diện phân tích dữ liệu và mô hình hóa
│
├── train.py                       # Script chạy huấn luyện mô hình & kiểm định chéo 4 quý
├── predict.py                     # Script sinh file dự báo submission.csv
├── inspect_model.py               # Script xem chi tiết nội dung các file models (.pkl)
├── submission.csv                 # File nộp bài Kaggle chính thức
└── README.md                      # Hướng dẫn chi tiết dự án
```

---

## 2. Hướng Dẫn Sử Dụng (Quick Start)

### Huấn Luyện & Kiểm Định Chéo 4 Quý (Train)
Chạy lệnh sau trong terminal:
```bash
python train.py
```
*Script sẽ kiểm định mô hình qua 4 quý năm 2025 (Q1, Q2, Q3, Q4), in bảng so sánh chi tiết và lưu mô hình vào thư mục `models/`.*

### Sinh File Dự Báo Nộp Bài (Predict)
Chạy lệnh sau trong terminal:
```bash
python predict.py
```
*Script sẽ nạp mô hình từ `models/`, tính toán dự báo cho toàn bộ 3.869 dòng của Quý 1/2026 và lưu kết quả vào `submissions/submission.csv` (đồng thời cập nhật `submission.csv` ở thư mục gốc).*

### Xem Nội Dung Bên Trong Các File Models (Inspect)
Chạy lệnh sau trong terminal:
```bash
python inspect_model.py
```
*Script sẽ đọc các file `models/*.pkl` và hiển thị chi tiết các hệ số khuyến mãi, tác động thời tiết, doanh thu cơ sở từng quán.*

---

## 3. Hiệu Năng So Với Baseline

| Mô Hình | Q1/2025 | Q2/2025 | Q3/2025 | Q4/2025 | Trung Bình 4 Quý | Cải Thiện (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Gốc (Chưa lọc lỗi)** | 0.2334 | 0.2366 | 0.2364 | 0.2387 | **0.2363** | — |
| **Baseline Đã Làm Sạch** | 0.1983 | 0.2016 | 0.2023 | 0.2052 | **0.2018** | +14.6% |
| **CafeRevenueModel (Tối ưu)** | **0.1045** | **0.1072** | **0.1017** | **0.1117** | **0.1063** | **+47.3%** |

---

## 4. Đặc Điểm Kỹ Thuật Chính

1. **Làm sạch nhiễu DGP**: Tự động sửa lỗi nhập số dư thừa (1000x, 10x, 0.1x), chuẩn hóa nhiệt độ từ độ F về độ C, sửa lỗi tên quận và điền diện tích theo tỷ lệ mặt bằng.
2. **Kiến trúc Hierarchical Multi-Stage**: Tách biệt hiệu ứng toàn hệ thống (Shared Multiplicative Effects: Khuyến mãi, Thời tiết mưa/nắng, Hiệu ứng thứ trong tuần theo loại quán, Phân kỳ chuỗi ngày Tết) và độ lệch cơ sở riêng từng quán (Store-specific base intercept).
3. **Nội suy quán mới (Cold-Start)**: Sử dụng mô hình hồi quy Empirical Bayes dựa trên diện tích, chỗ ngồi, loại quán và bãi đỗ xe để ước lượng doanh thu cho các quán mới mở trước khi đủ lịch sử.
4. **Hiệu chỉnh chuỗi thời gian AR(1)**: Dự báo độ trôi ngắn hạn của từng cửa hàng theo hàm suy giảm $0.80^h$, cải thiện mạnh mẽ độ chính xác những tuần đầu tiên.
5. **Stage-2 GBDT Residual Refinement**: Tăng cường phi tuyến qua `HistGradientBoostingRegressor` học lại phần thặng dư chưa giải thích.
