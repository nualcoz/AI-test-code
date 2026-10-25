# Báo Cáo Phân Tích Dữ Liệu & Giải Pháp Mô Hình Hóa Dự Báo Rời Bỏ Thẻ (Card Churn)

Tài liệu này tổng hợp toàn diện phân tích dữ liệu chuyên sâu (EDA), kiểm toán chất lượng dữ liệu (Data Quality Audit), kỹ thuật đặc trưng nghiệp vụ ngân hàng và kiến trúc mô hình học máy tối ưu cho bài toán **Card Churn Prediction**.

---

## 1. Tổng Quan Bài Toán & Thang Đo Đánh Giá

- **Nhiệm vụ chính**: Dự đoán xác suất khách hàng đóng thẻ tín dụng (`churn_3m = 1`) trong vòng 3 tháng kể từ ngày chốt dữ liệu (**30/06/2026**).
- **Thang đo đánh giá**: **ROC-AUC** (Receiver Operating Characteristic - Area Under Curve).
  - Thang đo chỉ đánh giá **thứ tự xếp hạng xác suất** (khách hàng có nguy cơ rời bỏ cao có được dự đoán xác suất lớn hơn khách hàng ở lại hay không).
  - Không bị ảnh hưởng bởi việc mất cân bằng lớp (Class Imbalance).
- **Tập dữ liệu**:
  - `train.csv`: 12.120 bản ghi (gồm 120 dòng trùng lặp, còn 12.000 khách hàng duy nhất), 24 cột.
  - Tỷ lệ churn tự nhiên: **11.66%** (1.400 khách hàng rời bỏ / 12.000 khách hàng).
  - `sample_submission.csv`: 8.000 khách hàng cần dự báo (`customer_id`, `churn_3m`).

---

## 2. Kiểm Toán Dữ Liệu Bẩn & Các Bất Thường Phát Hiện

Qua quá trình kiểm toán chuyên sâu, chúng tôi phát hiện 5 nhóm lỗi nhập liệu và dị biệt mang tính hệ thống:

```
+-------------------------------------------------------------------------------------------------------+
|                               KIỂM TOÁN CHẤT LƯỢNG DỮ LIỆU CARD CHURN                                 |
+-------------------------------------------------------------------------------------------------------+
| 1. Dòng trùng lặp (Duplicates)          : 120 dòng nhân bản toàn bộ -> Loại bỏ để tránh data leakage  |
| 2. Lỗi đơn vị tiền tệ (Unit Error x1e6) : 351 dòng thu nhập & hạn mức nhập VND thay vì triệu VND      |
| 3. Giá trị Sentinel ở Tuổi (Age)        : 69 dòng age < 18 (0, 1) hoặc age > 90 (150, 200, 999)       |
| 4. Giá trị Sentinel ở Sử dụng (Util)    : 79 dòng utilization_6m < 0 (-1.0) hoặc > 2.0 (9.99, 15.0)   |
| 5. Giá trị Số âm ở Giao dịch (Txn Amt)  : 169 dòng txn_amount_3m < 0 (-2.99 đến -0.12) do hoàn tiền   |
| 6. Đa dạng hóa văn bản (Text variants)  : gender (8 kiểu), province (41 kiểu), occupation (15 kiểu)  |
+-------------------------------------------------------------------------------------------------------+
```

### Chi tiết các lỗi dữ liệu & Giải pháp chuẩn hóa:
1. **Lỗi đơn vị tiền tệ ($\times 10^6$)**:
   - Trong khi phần lớn `income_month` có trung vị 17.7 triệu VNĐ và `credit_limit` trung vị 40.0 triệu VNĐ, có 351 dòng có giá trị $\ge 100.000$ (ví dụ: `25.000.000` VND hoặc `100.000.000` VND).
   - **Giải pháp**: Nhận diện ngưỡng $\ge 100.000$ và tự động chia cho $10^6$ để đưa về đơn vị chuẩn (triệu VNĐ).
2. **Lỗi giá trị Sentinel ở Tuổi (`age`)**:
   - Xuất hiện các giá trị sentinel phi thực tế như `0, 1` (trẻ sơ sinh) hoặc `150, 200, 999` (mã quy ước lỗi).
   - **Giải pháp**: Gán thành `NaN` và suy luận chéo theo nghề nghiệp (`student` $\rightarrow 21$ tuổi, `retired` $\rightarrow 66$ tuổi, các ngành khác $\rightarrow 37$ tuổi).
3. **Lỗi hạn mức tín dụng âm và vượt trần (`utilization_6m`)**:
   - `utilization_6m` có các giá trị `-1.0, -0.5` hoặc `9.99, 15.0`.
   - **Giải pháp**: Gán các giá trị ngoài đoạn $[0, 2.0]$ về `NaN` và điền bằng trung vị chuẩn.
4. **Lỗi số tiền giao dịch âm (`txn_amount_3m`)**:
   - Các giá trị âm từ `-2.99` đến `-0.12` thường xuất hiện khi khách hàng được hoàn tiền (refund/chargeback) vượt quá chi tiêu.
   - **Giải pháp**: Ước lượng lại bằng công thức nghiệp vụ: `avg_txn_ratio * income * txn_count`.

---

## 3. Khám Phá Nghiệp Vụ Ngân Hàng & Tương Tác Phi Tuyến

Phân tích tương quan và trực quan hóa dữ liệu đã bộc lộ 3 cơ chế rời bỏ thẻ tín dụng cốt lõi:

### 1. Hiệu Ứng Vách Đá Phí Thường Niên (Annual Fee Anniversary Cliff)
- Thẻ tín dụng thường thu phí thường niên sau mỗi 12 tháng kể từ ngày mở thẻ (`open_date`).
- Khách hàng có tháng mở thẻ rơi vào **Tháng 7, 8, 9** (kỷ niệm 1 năm, 2 năm mở thẻ ngay trong 3 tháng tới) mà **không được miễn phí thường niên** (`annual_fee_waived == 0`) có tỷ lệ rời bỏ tăng vọt lên **25.1%** (so với mức trung bình 11.7%).
- Tỷ số nguy cơ (Odds Ratio) trong mô hình Logistic GAM đạt **x1.61**.

### 2. Hiệu Ứng Khách Hàng Săn Khuyến Mãi Thẻ Hoàn Tiền (Cashback Promo Hunter)
- Nhóm khách hàng mở thẻ **Cashback** và vừa kết thúc giai đoạn miễn lãi/ưu đãi mở thẻ trong 60 ngày qua (`promo_ended_60d == 1`) có tỷ lệ đóng thẻ lên tới **24.2%**.
- Đây là nhóm khách hàng săn ưu đãi ngắn hạn (churner theo vụ mùa khuyến mại).

### 3. Tín Hiệu Rời Bỏ Từ Kênh Số (Digital Inactivity Signal)
- Khách hàng không có lượt đăng nhập ứng dụng (`app_logins_30d = NaN`) có tỷ lệ rời bỏ lên tới **17.4%** (so với 10.9% ở nhóm có đăng nhập). Việc tạo biến cờ hiệu `app_logins_is_na` mang lại tín hiệu dự báo phân tách rõ nét.

---

## 4. Kiến Trúc Mô Hình Học Máy & Kết Quả Kiểm Định Chéo 5-Fold

Mô hình tối ưu được xây dựng theo kiến trúc **Hybrid Ensemble kết hợp 2 nhánh**:
1. **Multi-seed HistGradientBoostingClassifier (5 seeds)**: Bắt các tương tác phi tuyến rời rạc, tự nhiên xử lý các giá trị khuyết thiếu, điều chuẩn $L_2 = 10.0$ và giới hạn lá `max_leaf_nodes = 6` để triệt tiêu overfit.
2. **Spline GAM LogisticRegression**: Mô hình cộng tính tổng quát với phép biến đổi B-spline bậc 3 cho các biến số liên tục (`age`, `income`, `limit`, `utilization`, `spend_change_pct`, `tenure_days`) kết hợp biến tương tác nghiệp vụ.

### Bảng Kết Quả Kiểm Định Chéo 5-Fold Stratified CV (ROC-AUC)

| Giai Đoạn Mô Hình | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | OOF ROC-AUC | Mức Cải Thiện |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Giai đoạn 0: Baseline Thô (starter_churn)** | 0.7045 | 0.7418 | 0.7314 | 0.7378 | 0.7344 | **0.7302** | — |
| **Giai đoạn 1: Logistic Sau Làm Sạch** | 0.7138 | 0.7496 | 0.7533 | 0.7364 | 0.7418 | **0.7389** | +0.87% |
| **Giai đoạn 2: Spline GAM + Tương Tác** | 0.7820 | 0.8282 | 0.8187 | 0.8163 | 0.8053 | **0.8096** | +7.94% |
| **Giai đoạn 3: Multi-Seed HGB (5 seeds)** | 0.7831 | 0.8255 | 0.8139 | 0.8163 | 0.8098 | **0.8093** | +7.91% |
| **Giai đoạn 4: Hybrid Ensemble (HGB + GAM)** | **0.7860** | **0.8304** | **0.8195** | **0.8192** | **0.8117** | **0.8129** | **+8.27%** |

---

## 5. Kết Luận & Hướng Dẫn Nộp Bài

1. Toàn bộ quy trình xử lý, làm sạch và mô hình hóa đã được đóng gói thành một `ChurnPipeline` khép kín.
2. File kết quả kiểm thử đã được lưu trữ sẵn sàng tại [submissions/submission.csv](file:///E:/study/AI/test%201/Kaggle_Competition/data/churn/submissions/submission.csv) và [submission.csv](file:///E:/study/AI/test%201/Kaggle_Competition/data/churn/submission.csv).
3. Khi chạy trên Kaggle có `test.csv`, `predict.py` sẽ tự động phát hiện và sinh dự báo xác suất cá nhân hóa cho từng khách hàng.
