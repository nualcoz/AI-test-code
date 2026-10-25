# Báo Cáo Phân Tích Dữ Liệu (EDA) — Dự Báo Doanh Thu Chuỗi Cafe

Tài liệu này tổng hợp kết quả phân tích khám phá dữ liệu (Exploratory Data Analysis) cho toàn bộ các bảng dữ liệu trong thư mục `data/cafe`. Toàn bộ mã nguồn phân tích và biểu đồ trực quan cũng đã được lưu và chạy sẵn trong notebook `phan_tich_du_lieu_cafe.ipynb`.

---

## 1. Tổng Quan Kiến Trúc Dữ Liệu (Data Schema & Overview)

| Bảng dữ liệu | Kích thước gốc | Khoảng thời gian | Khóa chính | Ghi chú quan trọng |
| :--- | :---: | :---: | :---: | :--- |
| `train.csv` | `26,632 × 4` | `2024-01-01` -> `2025-12-31` | `(date, store_id)` | Có 132 dòng trùng lặp -> còn `26,500` dòng duy nhất |
| `stores.csv` | `45 × 8` | Mở từ `2018` -> `12/2025` | `store_id` | Cửa hàng `S02` ngừng hoạt động sau ngày `14/08/2025` |
| `weather.csv` | `1,593 × 4` | `2024-01-01` -> `2026-03-31` | `(date, city)` | Thiếu 49 ngày (21 ngày ở HN, 28 ngày ở TP.HCM) và 16 ô `rain_mm` |
| `holidays.csv` | `55 × 3` | `2024-01-01` -> `2026-03-08` | `(date, type)` | `2024-02-14` và `2026-02-14` xuất hiện 2 lần (Tết + Valentine) |
| `sample_submission.csv` | `3,869 × 2` | `2026-01-01` -> `2026-03-31` | `id` | 44 quán * 90 ngày - 55 (office nghỉ Tết) - 36 (uni nghỉ Tết) = 3,869 |

---

## 2. Kiểm Toán Chất Lượng Dữ Liệu & Phát Hiện Dị Biệt (Data Quality Audit)

Bộ dữ liệu được cài cắm có chủ đích **8 nhóm lỗi dữ liệu bẩn (messy data)**:

| STT | Bảng | Cột bị lỗi | Mô tả lỗi phát hiện | Số lượng | Phương pháp xử lý chuẩn |
| :-: | :--- | :--- | :--- | :---: | :--- |
| **1** | `train.csv` | `date, store_id` | Dòng trùng lặp hoàn toàn (`exact duplicates`) | `132` dòng | `drop_duplicates(subset=["date", "store_id"])` |
| **2** | `train.csv` | `revenue` | Cửa hàng `S12` tháng `03/2024` ghi theo đơn vị **nghìn VNĐ** (`10,620` – `53,600`) thay vì triệu VNĐ | `30` ngày | Chia `revenue` cho `1000` khi `revenue > 1000` |
| **3** | `train.csv` | `revenue` | Lỗi dịch dấu thập phân sang phải (**gấp 10x**, ví dụ `275.60` thay vì `27.56`) | `53` dòng | Chia `revenue` cho `10` khi `days_since_open > 14` và `revenue / med_dow > 3.0` |
| **4** | `train.csv` | `revenue` | Lỗi dịch dấu thập phân sang trái (**bằng 0.1x**, ví dụ `0.57` thay vì `5.70`, `1.21` thay vì `12.10`) | `26` dòng | Nhân `revenue` với `10` khi `revenue / med_dow < 0.20` |
| **5** | `stores.csv` | `store_type`, `district`, `open_date` | Chữ hoa/thường lộn xộn (`Mall`/`mall`); Quận 10 ghi thành `District 10` và `Q.10`; `open_date` trộn lẫn `YYYY-MM-DD` và `DD/MM/YYYY` | Toàn bảng | `str.lower()`, gộp `Q.10 -> District 10`, `pd.to_datetime(..., format="mixed", dayfirst=True)` |
| **6** | `stores.csv` | `area_m2` | Khuyết diện tích (`NaN`) tại `S19` (57 ghế, `street`) và `S24` (55 ghế, `office`) | `2` quán | Nội suy từ `seats` * tỷ lệ trung vị `area_m2 / seats` theo `store_type` (`145.2 m²` và `162.1 m²`) |
| **7** | `weather.csv` | `temp_c` | Nhiệt độ TP.HCM từ `10/06/2025` đến `01/07/2025` bị ghi theo **độ Fahrenheit (`82.2–89.2 °F`)** | `22` dòng | Khi `temp_c > 45`, quy đổi `C = (F - 32) * 5 / 9` (`27.9–31.8 °C`) |
| **8** | `weather.csv` & `holidays.csv` | `date`, `rain_mm` | `weather.csv` thiếu 49 ngày và 16 giá trị `rain_mm`; `holidays.csv` lặp 2 dòng vào ngày `14/02/2024` và `14/02/2026` | `49` ngày & `2` ngày lễ | Dựng lưới ngày đầy đủ, nội suy tuyến tính `temp_c`, điền `0` cho `rain_mm`; tách cột `nat_holiday` và `evt_holiday` |

---

## 3. Phân Tích Hệ Thống Cửa Hàng & Chu Kỳ Thứ Trong Tuần

| Loại cửa hàng (`store_type`) | Thành phố | Số lượng quán | Diện tích TB (`m²`) | Số ghế TB (`seats`) | Doanh thu ngày thường TB (triệu VNĐ) | Đặc trưng khách hàng |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`mall`** | Hà Nội / TP.HCM | `4` / `4` | `98.2` / `98.0` | `33.2` / `38.0` | `18.96` / `25.58` | Doanh thu/m² cao nhất, tăng mạnh cuối tuần & ngày mưa |
| **`street`** | Hà Nội / TP.HCM | `10` / `5` | `156.5` / `177.0` | `62.3` / `67.4` | `21.74` / `23.07` | Diện tích rộng, đông khách cuối tuần, nhạy cảm với mưa lớn |
| **`office`** | Hà Nội / TP.HCM | `5` / `6` | `126.8` / `114.8` | `47.4` / `39.5` | `17.63` / `16.16` | Tập trung khách văn phòng T2–T6, vắng khách T7–CN & ngày lễ |
| **`university`** | Hà Nội / TP.HCM | `3` / `3` | `170.0` / `183.0` | `64.7` / `61.3` | `13.83` / `17.61` | Diện tích lớn nhưng giá bình dân; sụt giảm mạnh dịp Hè & Tết |
| **`residential`** | Hà Nội / TP.HCM | `3` / `2` | `79.7` / `75.0` | `27.3` / `35.0` | `7.16` / `12.94` | Quy mô nhỏ phục vụ khu dân cư, tăng nhẹ vào cuối tuần |

---

## 4. Vòng Đời Cửa Hàng Mới Mở (`days_since_open`)

1. **Giai đoạn Khai trương (`days_since_open` in {0, 1, 2})**: Trong 3 ngày đầu mở bán, doanh thu cao hơn mức trưởng thành từ **`+30%` đến `+46%`** trên thang log.
2. **Giai đoạn Chạy đà 60 ngày (`days_since_open` in [3, 62])**:
   - Từ ngày thứ 3, doanh thu bắt đầu ở mức **`~55%` công suất trưởng thành** (chênh lệch `-0.60` trên thang log).
   - Trong 60 ngày tiếp theo, doanh thu tăng tuyến tính đều đặn (`+0.01` log-revenue mỗi ngày) cho tới khi đạt `100%` công suất trưởng thành từ ngày thứ `63` trở đi.

---

## 5. Phân Tích Tết Nguyên Đán (`tet_offset`) & Các Ngày Lễ (`holidays.csv`)

| Giai đoạn Tết (`tet_offset` so với Mùng 1) | `mall` | `street` | `residential` | `office` | `university` | Giải thích hành vi thực tế |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`[-14, -13]` (16–17 tháng Chạp)** | `0.00` | `+0.08` | `+0.08` | `+0.09` | **`-0.50`** | Sinh viên đại học bắt đầu nghỉ Tết về quê sớm |
| **`[-12, -8]` (18–22 tháng Chạp)** | **`+0.18`** | `+0.06` | `+0.06` | `+0.03` | **`-0.63`** | Nhu cầu mua sắm cuối năm tại trung tâm thương mại tăng mạnh |
| **`[-7]` (23 Tết — Ông Công Ông Táo)** | **`+0.39`** | **`+0.29`** | **`+0.15`** | `+0.07` | **`-0.55`** | Đỉnh mua sắm & đi lại trước Tết |
| **`[-6, -3]` (24–27 tháng Chạp)** | **`+0.15`** | **`+0.14`** | `+0.09` | `+0.02` | **`-0.58`** | Các quán phố và trung tâm thương mại vẫn đông khách |
| **`[-2, +1]` (28 Tết -> Mùng 2 Tết)** | **`-0.60`** | **`-0.96`** | **`-0.88`** | *Đóng cửa* | *Đóng cửa* | Khối `office` & `university` đóng cửa; các quán mở xuyên Tết vắng khách |
| **`[+2]` (Mùng 3 Tết)** | **`+0.04`** | **`-0.92`** | **`-0.89`** | *Đóng cửa* | *Đóng cửa* | `mall` phục hồi ngay lập tức nhờ khách đi chơi xuân trong TTTM |
| **`[+3, +7]` (Mùng 4 -> Mùng 8 Tết)** | `+0.02` | `-0.21` | `-0.24` | **`-0.45`** | **`-0.56`** | Công sở mở lại nhưng nhiều người còn nghỉ phép; Mùng 4 `university` vẫn đóng |
| **`[+8, +9]` (Mùng 9 -> Mùng 10 Tết)** | `-0.14` | `-0.01` | `+0.02` | **`-0.43`** | **`-0.62`** | `street` và `residential` trở lại bình thường; `office` vẫn thấp |
| **`[+10, +12]` (Mùng 11 -> Mùng 13 Tết)** | `-0.09` | `0.00` | `-0.03` | `+0.02` | **`-0.56`** | `office` bình thường trở lại; sinh viên đại học hết ngày `+12` mới đi học lại |

---

## 6. Tính Mùa Vụ (Kỳ Nghỉ Hè Đại Học) & Tác Động Của Thời Tiết

1. **Kỳ nghỉ hè khối `university` (`Tháng 7 & Tháng 8`)**: Toàn bộ 6 quán `university` sụt giảm trung bình **`-0.415` log-revenue** (`~34%` doanh thu) từ ngày `01/07` đến `31/08` do sinh viên nghỉ hè, và phục hồi ngay từ ngày `04/09`.
2. **Phản ứng ngược chiều với lượng mưa (`rain_mm`)**: Khi mưa to (`> 10 mm`), doanh thu các quán `street`, `university` và `residential` giảm sâu từ **`-8%` đến `-41%`** (thang log), trong khi các quán trong **`mall` tăng nhẹ `+2.5%` đến `+3.7%`**.
3. **Phản ứng với nhiệt độ (`temp_c`)**: Tại Hà Nội, những ngày mùa đông rét đậm (`< 17 °C`) làm doanh thu quán `street` và `residential` giảm `-5%` đến `-7%`; tại TP.HCM, những ngày nắng nóng (`> 29.5 °C`) giúp doanh thu tăng `+3%` đến `+6%`.
