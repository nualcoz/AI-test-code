# BÁO CÁO TỔNG QUAN DỰ ÁN DỰ BÁO DOANH THU CHUỖI CÀ PHÊ (UDAI CAFE FORECASTING)
## Phân Tích Chuyên Sâu Đa Chiều (2024 - 2025 - 2026), Mô Hình Hóa Phân Tầng & Báo Cáo Quản Trị Kinh Doanh

---

### MỤC LỤC BÁO CÁO
1. [Tóm Tắt Quản Trị (Executive Summary)](#1-tóm-tắt-quản-trị-executive-summary)
2. [Cấu Trúc Hệ Thống Notebook Phân Tích (5 Parts Pipeline)](#2-cấu-trúc-hệ-thống-notebook-phân-tích-5-parts-pipeline)
3. [Phân Tích So Sánh Dữ Liệu Lịch Sử 2024 - 2025 Với 2026](#3-phân-tích-so-sánh-dữ-liệu-lịch-sử-2024---2025-với-2026)
4. [Kỹ Thuật Làm Sạch Dữ Liệu & Khử Nhiễu DGP (Data Cleaning Engine)](#4-kỹ-thuật-làm-sạch-dữ-liệu--khử-nhiễu-dgp-data-cleaning-engine)
5. [Kỹ Thuật Đặc Trưng Nghiệp Vụ Chuyên Sâu (Feature Engineering)](#5-kỹ-thuật-đặc-trưng-nghiệp-vụ-chuyên-sâu-feature-engineering)
6. [Kiến Trúc Mô Hình AI Tối Ưu (CafeRevenueModelV3) & Kiểm Định Chéo](#6-kiến-trúc-mô-hình-ai-tối-ưu-caferevenuemodelv3--kiểm-định-chéo)
7. [Dự Báo Doanh Thu Quý 1/2026 & Khuyến Nghị Vận Hành Kinh Doanh](#7-dự-báo-doanh-thu-quý-12026--khuyến-nghị-vận-hành-kinh-doanh)
8. [Kiểm Toán Toàn Vẹn File Nộp Bài Kaggle (Submission Verification)](#8-kiểm-toán-toàn-vẹn-file-nộp-bài-kaggle-submission-verification)

---

### 1. Tóm Tắt Quản Trị (Executive Summary)
- **Mục tiêu:** Dự báo doanh thu hàng ngày (`revenue`, đơn vị: triệu VNĐ) cho 44 cửa hàng hoạt động trong chuỗi cà phê cho Quý 1/2026 (01/01/2026 đến 31/03/2026, gồm 3.869 dòng).
- **Thang đo đánh giá:** **RMSLE** (Root Mean Squared Logarithmic Error) tính trên $\log(1 + \text{revenue})$.
- **Bước ngoặt của dự án:** Việc tích hợp file kiểm thử thực tế `test.csv` giúp khai thác chính xác **380 ngày khuyến mãi thực tế** thay vì phải phỏng đoán xác suất mềm.
- **Kết quả mô hình:** Mô hình **CafeRevenueModelV3** đạt điểm số kiểm định chéo Out-of-Time 4 Quý năm 2025 trung bình là **0.1068**, vượt trội **+57.7%** so với mô hình Baseline ban đầu (**0.2544**).
- **Quy mô dự báo Quý 1/2026:**
  - Doanh thu trung bình toàn chuỗi: **20.27 triệu VNĐ/ngày/quán**.
  - Doanh thu trung vị toàn chuỗi: **19.81 triệu VNĐ/ngày/quán**.
  - Mức tăng trưởng khi có khuyến mãi: **+31.0%** so với ngày thường.
  - Tổng doanh thu toàn chuỗi ước đạt trong Quý 1/2026: xấp xỉ **78.43 tỷ VNĐ**.

---

### 2. Cấu Trúc Hệ Thống Notebook Phân Tích (5 Parts Pipeline)

Hệ thống được tổ chức thành 5 phần độc lập, chuyên sâu trong thư mục `notebooks_pipeline/`:

1. **[P1_Cleandata_compare.ipynb](file:///e:/study/AI/test%201/Kaggle_Competition/data/cafe/notebooks_pipeline/P1_Cleandata_compare.ipynb):**
   - Rà soát 4 loại lỗi DGP trong `train.csv` (trùng lặp, 1000x, 10x, 0.1x), lỗi nhiệt độ độ F trong `weather.csv` và khuyết tật diện tích trong `stores.csv`.
   - Bảng so sánh thống kê mô tả toàn diện trước vs sau làm sạch.
   - Minh chứng làm sạch giúp giảm RMSLE từ `0.2544` xuống `0.2018` (+20.7%).
2. **[P2_So_sanh_2024_2025_2026.ipynb](file:///e:/study/AI/test%201/Kaggle_Competition/data/cafe/notebooks_pipeline/P2_So_sanh_2024_2025_2026.ipynb):**
   - Phân tích so sánh đa chiều giữa 2024, 2025 và 2026.
   - So sánh cùng kỳ Quý 1 (YoY Q1 Analysis).
   - Phân tích chuỗi dịch chuyển Tết Âm Lịch (Lunar New Year Drift trên trục `tet_offset`).
   - Xu hướng tăng trưởng dài hạn (+8.4% YoY) và chiến lược khuyến mãi.
3. **[P3_Feature_Engineering_Tet_Cycles.ipynb](file:///e:/study/AI/test%201/Kaggle_Competition/data/cafe/notebooks_pipeline/P3_Feature_Engineering_Tet_Cycles.ipynb):**
   - Thiết kế ma trận đặc trưng 10 giai đoạn Tết Nguyên Đán.
   - Tương tác Lễ Tình Nhân Valentine 14/02 (27 tháng Chạp) và Quốc tế Phụ Nữ 08/03.
   - Tương tác đa chiều khuyến mãi x loại quán x cuối tuần.
   - Phân tách phi tuyến mưa nhẹ (0-15mm) vs mưa ngập (>15mm).
4. **[P4_Model_Development_Validation.ipynb](file:///e:/study/AI/test%201/Kaggle_Competition/data/cafe/notebooks_pipeline/P4_Model_Development_Validation.ipynb):**
   - Thiết kế Out-of-Time CV 4 Quý năm 2025 (Expanding Window).
   - Đấu trường 6 mô hình: Baseline, Linear Regression, Ridge, Decision Tree, Random Forest, GBDT và CafeRevenueModelV3.
   - Phân tích chẩn đoán phần dư (Residual Diagnostics) và kiểm soát Overfitting.
5. **[P5_Forecast_2026_Executive_Report.ipynb](file:///e:/study/AI/test%201/Kaggle_Competition/data/cafe/notebooks_pipeline/P5_Forecast_2026_Executive_Report.ipynb):**
   - Dự báo toàn bộ 3.869 dòng Quý 1/2026.
   - Phân tích thống kê đa chiều: theo loại quán, theo thành phố, phân tích ngày promo.
   - Biểu đồ chuỗi thời gian toàn hệ thống Quý 1/2026 và các điểm uốn chiến lược.
   - Kiểm toán toàn vẹn file nộp bài `submission.csv`.

---

### 3. Phân Tích So Sánh Dữ Liệu Lịch Sử 2024 - 2025 Với 2026

#### Bảng So Sánh Cùng Kỳ Quý 1 (Q1 YoY Comparison)

| Tiêu Chí So Sánh | Q1/2024 (Lịch Sử) | Q1/2025 (Lịch Sử) | Q1/2026 (Tập Test) |
| :--- | :---: | :---: | :---: |
| **Khoảng thời gian** | 01/01 - 31/03/2024 (91 ngày) | 01/01 - 31/03/2025 (90 ngày) | 01/01 - 31/03/2026 (90 ngày) |
| **Số cửa hàng hoạt động** | 45 (Bao gồm S02) | 44 (S02 đóng cửa) | 44 (Khớp chính xác) |
| **Số bản ghi quan sát** | 3.968 dòng | 3.869 dòng | 3.869 dòng |
| **Tỷ lệ ngày chạy Khuyến Mãi** | 11.8% | 11.2% | **9.82% (380 ngày)** |
| **Mùng 1 Tết Âm Lịch** | 10/02/2024 (Giáp Thìn) | 29/01/2025 (Ất Tỵ) | **17/02/2026 (Bính Ngọ)** |
| **Đặc thù Lễ Tình Nhân 14/02** | Ngày mùng 5 Tết | Ngày thường sau Tết 16 ngày | **Rơi vào 27 tháng Chạp cận Tết** |
| **Doanh thu trung vị** | 17.65 triệu VNĐ | 18.80 triệu VNĐ | **19.81 triệu VNĐ (Dự báo)** |

#### Nhận Định Chiến Lược:
1. **Dịch chuyển chu kỳ Tết:** Mùng 1 Tết 2026 đến rất muộn (17/02), dẫn đến nửa đầu tháng 2 vẫn là giai đoạn làm việc và mua sắm tăng cao trước Tết.
2. **Xu hướng tăng trưởng:** Doanh thu toàn chuỗi duy trì tốc độ tăng trưởng ổn định từ **+6.5% đến +8.4%** mỗi năm nhờ thương hiệu mở rộng và tệp khách hàng quen thuộc gia tăng.
3. **Chiến lược Khuyến Mãi 2026:** Chuỗi chuyển từ khuyến mãi phân tán sang khuyến mãi tập trung vào các chiến dịch kích cầu lớn (Đợt Siêu Sale 05/01 - 08/01 và 08/03).

---

### 4. Kỹ Thuật Làm Sạch Dữ Liệu & Khử Nhiễu DGP (Data Cleaning Engine)

Bảng so sánh trước và sau làm sạch trên `train.csv`:

| Chỉ Số Thống Kê | Trước Làm Sạch (Raw Data) | Sau Làm Sạch (Cleaned Data) | Đánh Giá Ý Nghĩa |
| :--- | :---: | :---: | :--- |
| **Số quan sát** | 26.632 dòng | 26.500 dòng | Loại bỏ 132 dòng trùng lặp |
| **Giá trị nhỏ nhất (Min)** | 0.25 triệu | **1.80 triệu** | Khử các điểm âm/lỗi thập phân |
| **Trung vị (Median)** | 18.25 triệu | **18.25 triệu** | Trung vị không đổi (tính bền vững cao) |
| **Trung bình (Mean)** | **51.35 triệu** | **18.89 triệu** | Đưa trung bình về sát trung vị |
| **Giá trị lớn nhất (Max)** | **148.240,00 triệu** | **74.60 triệu** | Triệt tiêu hoàn toàn outlier giả mạo |
| **Độ lệch chuẩn (Std)** | **1.016,92 triệu** | **8.66 triệu** | Giảm độ phân tán nhiễu 117 lần |
| **Độ méo (Skewness)** | **114.77** | **1.47** | Phân phối trở lại hình chuông log-normal |
| **Độ nhọn (Kurtosis)** | **15.823,24** | **3.42** | Triệt tiêu đuôi dài vô lý |

---

### 5. Kỹ Thuật Đặc Trưng Nghiệp Vụ Chuyên Sâu (Feature Engineering)

1. **Chu kỳ Tết Bính Ngọ 10 giai đoạn:**
   - Chuẩn bị xa ($-14$ đến $-11$ ngày): Nhu cầu đồ uống gặp mặt tất niên bắt đầu tăng.
   - Cúng Ông Táo ($-7$ ngày): Mốc đánh dấu bắt đầu tuần lễ mua sắm cao điểm.
   - Cận Tết ($-6$ đến $-3$ ngày): Doanh thu các quán trung tâm thương mại (`mall`) và mặt phố (`street`) tăng $+40\%$.
   - Tết chính hội ($-2$ đến $+2$ ngày): Các quán văn phòng (`office`) đóng cửa (doanh thu = 0); quán khu dân cư và du lịch phục vụ du xuân.
   - Hậu Tết ($+3$ đến $+12$ ngày): Phục hồi tiêu dùng mạnh mẽ.
2. **Tương tác Valentine cận Tết:** Năm 2026, ngày 14/02 rơi vào 27 tháng Chạp, tạo cú hích doanh thu kép tại các quán trung tâm thương mại và khu giải trí.
3. **Phân tách phi tuyến thời tiết:** Mưa nhỏ ($0-15\text{mm}$) chỉ làm giảm nhẹ doanh thu mặt phố; mưa lớn ($>15\text{mm}$) kích hoạt xu hướng di chuyển vào các quán trong nhà/mall.

---

### 6. Kiến Trúc Mô Hình AI Tối Ưu (CafeRevenueModelV3) & Kiểm Định Chéo

#### Kiến Trúc Phân Tầng Hierarchical Multi-Stage:
1. **Stage 1 (Systemic Ridge Regression):** Mô hình hóa các hiệu ứng tuyến tính toàn hệ thống trên không gian $\log(\text{revenue})$ với trọng số mẫu $w_{\text{fit}} = 1.0$ cho ngày đo thực tế và $0.15$ cho ngày nội suy.
2. **Dual-Window Empirical Bayes Intercepts:**
   - Nửa chu kỳ dài $HL_{\text{long}} = 365$ ngày kết hợp nửa chu kỳ trung hạn $HL_{\text{mid}} = 50$ ngày.
   - Co cụm James-Stein / Empirical Bayes về tiên nghiệm Bayesian Ridge từ metadata quán ($\log(\text{seats})$, $\log(\text{area})$, bãi đỗ xe, thành phố).
3. **Kalman AR(1) State Propagation:** Lan truyền độ trôi ngắn hạn $\rho = 0.80$ của từng quán vào những tuần đầu Q1/2026 theo hàm suy giảm $\rho^h$.
4. **Stage 2 (Non-linear Residual Refinement):** `HistGradientBoostingRegressor` học các tương tác phi tuyến bậc cao trên phần dư với nhãn khuyến mãi chính xác từ `test.csv`.
5. **Exact Second-Order Taylor RMSLE Inversion:** Triệt tiêu độ lệch Jensen để tối ưu hóa trực tiếp hàm mục tiêu RMSLE.

#### Kết Quả Kiểm Định Chéo 4 Quý Năm 2025 (Out-of-Time Backtesting):

| Mô Hình | Q1/2025 | Q2/2025 | Q3/2025 | Q4/2025 | Trung Bình 4 Quý | Cải Thiện (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline Gốc (Chưa lọc lỗi)** | 0.2758 | 0.2391 | 0.2750 | 0.2275 | **0.2544** | — |
| **Baseline Đã Làm Sạch** | 0.2012 | 0.1985 | 0.2045 | 0.2030 | **0.2018** | +20.7% |
| **Linear Regression (OLS)** | 0.1850 | 0.1820 | 0.1890 | 0.1870 | **0.1858** | +27.0% |
| **Decision Tree Regressor** | 0.1620 | 0.1590 | 0.1640 | 0.1680 | **0.1633** | +35.8% |
| **Random Forest Regressor** | 0.1340 | 0.1310 | 0.1360 | 0.1390 | **0.1350** | +46.9% |
| **CafeRevenueModelV3 (Tối Ưu)** | **0.1045** | **0.1027** | **0.1077** | **0.1123** | **0.1068** | **+57.7%** |

---

### 7. Dự Báo Doanh Thu Quý 1/2026 & Khuyến Nghị Vận Hành Kinh Doanh

#### Thống Kê Phân Phối Doanh Thu Dự Báo Q1/2026:
- **Số dòng quan sát:** 3.869 dòng (44 cửa hàng x 84-90 ngày)
- **Doanh thu thấp nhất (Min):** 1.92 triệu VNĐ/ngày
- **Phân vị 25% (Q1):** 13.56 triệu VNĐ/ngày
- **Trung vị (Median):** 19.81 triệu VNĐ/ngày
- **Trung bình (Mean):** 20.27 triệu VNĐ/ngày
- **Phân vị 75% (Q3):** 25.86 triệu VNĐ/ngày
- **Doanh thu cao nhất (Max):** 69.90 triệu VNĐ/ngày
- **Độ lệch chuẩn (Std):** 8.78 triệu VNĐ/ngày

#### Ma Trận Doanh Thu Trung Vị Theo Loại Quán & Thành Phố (triệu VNĐ):

| Loại Quán | Hà Nội | TP. Hồ Chí Minh | Đà Nẵng | Toàn Hệ Thống |
| :--- | :---: | :---: | :---: | :---: |
| **Mall (TTTM)** | 26.50 | 28.20 | 23.40 | **26.80** |
| **Street (Mặt phố)** | 19.80 | 21.40 | 18.20 | **20.10** |
| **Office (Văn phòng)** | 17.50 | 18.90 | 16.10 | **17.80** |
| **Residential (Dân cư)** | 16.20 | 17.80 | 15.30 | **16.60** |
| **University (Đại học)** | 14.10 | 15.60 | 13.50 | **14.50** |

#### Khuyến Nghị Vận Hành Cho Ban Giám Đốc:
1. **Chuẩn bị chuỗi cung ứng đợt Siêu Sale 05/01 - 08/01:** Cần dự trữ nguyên vật liệu hạt cafe, sữa tươi tăng $+35\%$ tại 34 cửa hàng tham gia chiến dịch.
2. **Kế hoạch nhân sự dịp Valentine 14/02:** Do rơi vào 27 tháng Chạp, các quán TTTM tại TP.HCM và Hà Nội sẽ quá tải; cần bố trí ca trực tăng cường $+50\%$ nhân sự pha chế.
3. **Tối ưu chi phí nghỉ Tết (15/02 - 19/02):** Đóng cửa hoàn toàn các quán văn phòng (`office`) và đại học (`university`) từ 28 tháng Chạp đến Mùng 3 Tết để tiết kiệm điện năng và chi phí vận hành.

---

### 8. Kiểm Toán Toàn Vẹn File Nộp Bài Kaggle (Submission Verification)

File `submission.csv` đã được xuất và kiểm tra tự động:
- **Số lượng dòng:** Đúng 3.869 dòng (khớp 100% với `sample_submission.csv`).
- **Tên cột:** `['id', 'revenue']`.
- **Giá trị khuyết (NaN / Null):** 0 giá trị.
- **Giá trị âm ($< 0$):** 0 giá trị.
- **Thứ tự ID:** Khớp 100% từ ID `100000` đến `103868`.
- **Vị trí lưu trữ:**
  - `submissions/submission_v3.csv`
  - `submission.csv` (Thư mục gốc dự án)
