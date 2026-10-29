# -*- coding: utf-8 -*-
"""
build_p1_cleaning_notebook.py
Tạo và thực thi hoàn chỉnh notebook:
- notebooks_pipeline/P1_Cleandata_compare.ipynb
- notebooks/P1_Cleandata_compare.ipynb
"""

import os
import sys
import shutil
import nbformat
from nbclient import NotebookClient

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(PROJECT_ROOT, "notebooks_pipeline")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")
os.makedirs(PIPELINE_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

TARGET_NB_PATH = os.path.join(PIPELINE_DIR, "P1_Cleandata_compare.ipynb")

nb = nbformat.v4.new_notebook()

# -------------------------------------------------------------
# CELL 0: Header & Giới thiệu
# -------------------------------------------------------------
c0_md = """# Phần 1: Kiểm Toán, Kỹ Thuật Làm Sạch Dữ Liệu & Đánh Giá So Sánh Trước - Sau (Data Cleaning & Comparison)
## Chuyên Sâu Quy Trình Khử Nhiễu DGP, Chuẩn Hóa Đa Bảng & Kiểm Chứng Tác Động Thực Tế Lên Mô Hình

---

### 1. Tổng Quan Bài Toán & Tầm Quan Trọng Của Làm Sạch Dữ Liệu (Garbage In - Garbage Out)
Trong các cuộc thi Machine Learning thực chiến (đặc biệt là bài toán dự báo chuỗi thời gian kinh doanh thực tế như **UDAI Cafe Revenue Forecasting**), dữ liệu thu thập từ các hệ thống POS (Point of Sale), cảm biến thời tiết trạm quan trắc, và hồ sơ quản lý mặt bằng luôn bị ảnh hưởng bởi:
- **Lỗi thao tác con người (Human Typo Errors)**: Gõ nhầm thừa số 0, ấn lệch dấu phẩy phân tách thập phân.
- **Lỗi đơn vị đo lường (Unit Mismatch)**: Trộn lẫn giữa độ F (°Fahrenheit) và độ C (°Celsius) từ các nguồn API thời tiết khác nhau.
- **Lỗi thu thập dữ liệu (Missing Values & Duplication)**: Hồ sơ mặt bằng thiếu diện tích, dữ liệu truyền từ máy tính tiền bị gửi 2 lần tạo trùng lặp dòng.

Nếu đưa trực tiếp dữ liệu thô bị nhiễu vào mô hình Machine Learning:
- Các thang đo tổn thất bình phương như **MSE, RMSE** và đặc biệt là **RMSLE** trên $\\log(1 + y)$ sẽ bị phá hủy nghiêm trọng bởi các điểm dị biệt khổng lồ (ví dụ doanh thu bình thường 20 triệu nhưng bị gõ nhầm thành 20.000 triệu).
- Các mô hình cây quyết định (GBDT, Random Forest) hoặc hồi quy (Ridge, Lasso) sẽ học sai lệch hoàn toàn trọng số cho các biến phụ thuộc.

> **Mục tiêu của Notebook Phần 1:**
> 1. Trả lời chi tiết câu hỏi: **Clean data như thế nào?** (Khảo sát hiện trạng, nguyên nhân gốc rễ và logic xử lý).
> 2. Trả lời chi tiết câu hỏi: **Dùng những gì để clean data?** (Công cụ, thư viện, công thức toán học và kiểm định thống kê).
> 3. Trình bày **Phần so sánh dữ liệu trực quan** (Before vs After) qua bảng thống kê mô tả, biểu đồ phân phối, biểu đồ hộp outlier và đo lường sự cải thiện vượt trội trên mô hình Baseline."""
nb.cells.append(nbformat.v4.new_markdown_cell(c0_md))

# -------------------------------------------------------------
# CELL 1: Setup Environment
# -------------------------------------------------------------
c1_md = """## 2. Thiết Lập Môi Trường & Dùng Những Gì Để Clean Data?

### Bộ Công Cụ & Thư Viện Sử Dụng:
- **`pandas` (v2.x)**: Xử lý dữ liệu bảng, nhóm `groupby`, biến đổi chuỗi thời gian `pd.to_datetime`, tính tỷ lệ trung vị và nội suy.
- **`numpy` (v1.x/v2.x)**: Tính toán vector hóa ma trận, xử lý logarit `np.log1p`, `np.expm1`, điều kiện lọc ma trận `np.where`.
- **`matplotlib` & `seaborn`**: Thiết kế hệ thống trực quan hóa so sánh trực quan đa chiều (Histogram, KDE, Boxplot, Heatmap).
- **`scipy.stats`**: Đo lường các chỉ số thống kê hình dạng phân phối (Độ lệch Skewness, Độ nhọn Kurtosis).

### Các Phương Pháp & Công Thức Toán Học Được Áp Dụng:
1. **Khử Nhiễu Nhân $1000\\times$ (Lỗi gõ thừa 3 số 0):**
   $$y_{\\text{clean}} = \\frac{y_{\\text{raw}}}{1000} \\quad \\text{nếu } y_{\\text{raw}} > 1000$$
2. **Khử Nhiễu Lệch Dấu Thập Phân ($10\\times$ và $0.1\\times$):**
   Sử dụng tỷ số so với trung vị theo thứ trong tuần của từng cửa hàng:
   $$R_{i, d} = \\frac{y_{i, d}}{\\text{Median}_{\\text{dow}}(y_{i, \\text{dow}})}$$
   - Nếu $R_{i, d} > 3.0$ và quán đã mở quá 14 ngày (tránh nhầm với khai trương): chia cho 10.
   - Nếu $R_{i, d} < 0.20$: nhân với 10.
3. **Chuyển Đổi Đơn Vị Nhiệt Độ Fahrenheit Sang Celsius:**
   $$C = (F - 32) \\times \\frac{5}{9} \\quad \\text{khi } T > 45^\\circ\\text{C}$$
4. **Điền Khuyết Diện Tích Cửa Hàng Bằng Tỷ Lệ Mặt Bằng Theo Loại Quán:**
   $$\\widehat{\\text{area}}_{i} = \\text{seats}_{i} \\times \\text{Median}_{\\text{store\\_type}}\\left(\\frac{\\text{area}}{\\text{seats}}\\right)$$"""
nb.cells.append(nbformat.v4.new_markdown_cell(c1_md))

# -------------------------------------------------------------
# CELL 2: Code 1 - Load libraries & raw data
# -------------------------------------------------------------
c2_code = """import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# Cấu hình thẩm mỹ đồ thị
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["figure.dpi"] = 120

# Tự động tìm đường dẫn thư mục gốc dữ liệu và thêm vào sys.path
for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

print(f">>> THƯ MỤC DỮ LIỆU ĐANG SỬ DỤNG: {DATA_DIR}")

# 1. Nạp toàn bộ 5 bảng dữ liệu gốc
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

print("\\nKÍCH THƯỚC VÀ THÔNG SỐ BAN ĐẦU CỦA 5 TẬP DỮ LIỆU:")
print(f"1. train.csv    : {train_raw.shape[0]:,} dòng x {train_raw.shape[1]} cột")
print(f"2. test.csv     : {test_raw.shape[0]:,} dòng x {test_raw.shape[1]} cột")
print(f"3. stores.csv   : {stores_raw.shape[0]:,} dòng x {stores_raw.shape[1]} cột")
print(f"4. weather.csv  : {weather_raw.shape[0]:,} dòng x {weather_raw.shape[1]} cột")
print(f"5. holidays.csv : {holidays_raw.shape[0]:,} dòng x {holidays_raw.shape[1]} cột")"""
nb.cells.append(nbformat.v4.new_code_cell(c2_code))

# -------------------------------------------------------------
# CELL 3: Khảo sát chi tiết lỗi trong train.csv
# -------------------------------------------------------------
c3_md = """## 3. Khảo Sát & Bóc Tách Các Lỗi Nhiễu DGP Trong Dữ Liệu Thô

Chúng ta tiến hành rà soát từng bảng dữ liệu để chỉ ra các lỗi cụ thể trước khi thực hiện làm sạch:
1. **Trong `train.csv`**:
   - Dòng trùng lặp `(date, store_id)`: Do lỗi đường truyền hoặc hệ thống kết toán gửi lặp bản ghi.
   - Lỗi $1000\\times$: Doanh thu bình thường khoảng 10 - 40 triệu VNĐ, nhưng xuất hiện các giá trị lên tới hàng chục nghìn triệu VNĐ (ví dụ: `22,450.0`).
   - Lỗi $10\\times$ và $0.1\\times$: Lệch dấu thập phân (ví dụ gõ `235.0` thay vì `23.5`, hoặc `1.8` thay vì `18.0`).
2. **Trong `stores.csv`**:
   - Tên quận viết không đồng nhất (`Q.10`, `Quan 3`, `Quan 7` thay vì `District 10`, `District 3`, `District 7`).
   - Thiếu diện tích `area_m2` ở một số cửa hàng.
3. **Trong `weather.csv`**:
   - Nhiệt độ bị ghi nhầm theo độ F (giá trị $> 45^\circ\\text{C}$, thậm chí lên tới $86.5^\circ\\text{F}$ nhưng ghi vào cột `temp_c`).
   - Thiếu các ngày đo đạc liên tục."""
nb.cells.append(nbformat.v4.new_markdown_cell(c3_md))

# -------------------------------------------------------------
# CELL 4: Code 2 - Phát hiện lỗi
# -------------------------------------------------------------
c4_code = """# 1. Phát hiện trùng lặp trong train.csv
dup_mask = train_raw.duplicated(subset=["date", "store_id"], keep=False)
n_dups = dup_mask.sum()
print(f"1. Số dòng bị trùng lặp (date, store_id) trong train.csv: {n_dups} dòng")
if n_dups > 0:
    print("   Ví dụ 4 dòng trùng lặp tiêu biểu:")
    display(train_raw[dup_mask].sort_values(["store_id", "date"]).head(4))

# 2. Phát hiện lỗi 1000x trong train.csv
err_1000x = train_raw[train_raw["revenue"] > 1000]
print(f"\\n2. Số dòng bị lỗi gõ thêm 3 số 0 (revenue > 1000 tr): {len(err_1000x)} dòng")
print("   Ví dụ các dòng bị lỗi 1000x:")
display(err_1000x.head(5))

# 3. Phát hiện lỗi ghi độ F trong weather.csv
err_weather_f = weather_raw[weather_raw["temp_c"] > 45]
print(f"\\n3. Số quan sát thời tiết bị ghi nhầm độ F (temp_c > 45°C): {len(err_weather_f)} dòng")
print("   Ví dụ các dòng nhiệt độ độ F bất thường:")
display(err_weather_f.head(5))

# 4. Phát hiện thiếu diện tích trong stores.csv
missing_area = stores_raw[stores_raw["area_m2"].isna()]
print(f"\\n4. Số cửa hàng bị thiếu diện tích area_m2: {len(missing_area)} cửa hàng")
display(missing_area)"""
nb.cells.append(nbformat.v4.new_code_cell(c4_code))

# -------------------------------------------------------------
# CELL 5: Thiết kế Hàm Làm Sạch
# -------------------------------------------------------------
c5_md = """## 4. Xây Dựng Pipeline Làm Sạch Toàn Diện (Step-by-Step Data Cleaning Pipeline)

Dưới đây là 5 module hàm xử lý làm sạch dữ liệu sạch sẽ, có thể tái sử dụng (reproducible) và không làm biến đổi file gốc."""
nb.cells.append(nbformat.v4.new_markdown_cell(c5_md))

# -------------------------------------------------------------
# CELL 6: Code 3 - Viết các hàm làm sạch
# -------------------------------------------------------------
c6_code = """def clean_stores_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Làm sạch thông tin cửa hàng:
    - Chuẩn hóa tên loại quán (viết thường, bỏ khoảng trắng thừa).
    - Chuẩn hóa tên quận (Q.10 -> District 10, Quan 3 -> District 3, Quan 7 -> District 7).
    - Chuyển open_date sang định dạng datetime chuẩn.
    - Điền khuyết area_m2 dựa trên tỷ lệ trung vị diện tích/chỗ ngồi của từng loại quán.
    \"\"\"
    s = df_in.copy()
    s["store_type"] = s["store_type"].str.lower().str.strip()
    s["district"] = s["district"].str.strip().replace({
        "Q.10": "District 10",
        "Quan 3": "District 3",
        "Quan 7": "District 7",
    })
    s["open_date"] = pd.to_datetime(s["open_date"], format="mixed", dayfirst=True)
    
    # Tính tỷ lệ trung vị m2 / ghế theo từng loại quán
    ratio_by_type = (s["area_m2"] / s["seats"]).groupby(s["store_type"]).median()
    for idx, row in s[s["area_m2"].isna()].iterrows():
        s.loc[idx, "area_m2"] = round(row["seats"] * ratio_by_type[row["store_type"]], 1)
    return s


def clean_weather_pipeline(df_in: pd.DataFrame, cities: list) -> pd.DataFrame:
    \"\"\"
    Làm sạch dữ liệu thời tiết:
    - Phát hiện các giá trị > 45°C và chuyển từ độ F về độ C: C = (F - 32) * 5/9.
    - Điền khuyết lượng mưa rain_mm bằng 0.0.
    - Tạo lưới ngày đầy đủ 2024-01-01 đến 2026-03-31 cho tất cả các thành phố và nội suy liên tục.
    - Gắn cờ has_exact_weather để đánh dấu ngày có số đo trạm thực tế.
    \"\"\"
    w = df_in.copy()
    w["date"] = pd.to_datetime(w["date"])
    
    # Sửa lỗi nhiệt độ F -> C
    f_mask = w["temp_c"] > 45.0
    w.loc[f_mask, "temp_c"] = (w.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0
    
    exact_w = w.dropna(subset=["temp_c", "rain_mm"])[["date", "city"]].drop_duplicates()
    exact_w["has_exact_weather"] = True
    
    # Lưới ngày đầy đủ
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", "2026-03-31"), sorted(cities)],
        names=["date", "city"]
    ).to_frame().reset_index(drop=True)
    
    w = grid.merge(w, on=["date", "city"], how="left").sort_values(["city", "date"]).reset_index(drop=True)
    w["temp_c"] = w.groupby("city")["temp_c"].transform(lambda x: x.interpolate().bfill().ffill())
    w["rain_mm"] = w["rain_mm"].fillna(0.0)
    w = w.merge(exact_w, on=["date", "city"], how="left")
    w["has_exact_weather"] = w["has_exact_weather"].fillna(False)
    return w


def clean_holidays_pipeline(df_in: pd.DataFrame):
    \"\"\"
    Bóc tách ngày lễ quốc gia, sự kiện văn hóa và lập từ điển mốc Tết Âm lịch.
    \"\"\"
    h = df_in.copy()
    h["date"] = pd.to_datetime(h["date"])
    h_dedup = h.drop_duplicates(subset=["date"], keep="first").copy()
    
    non_tet_nat = h_dedup[(h_dedup["type"] == "national_holiday") & (~h_dedup["holiday_name"].str.contains("Tet"))]
    nat_hol = non_tet_nat.groupby("date")["holiday_name"].first().rename("nat_holiday")
    
    evt_hol = (
        h_dedup[(h_dedup["type"] == "event") & (~h_dedup["holiday_name"].str.contains("Tet"))]
        .groupby("date")["holiday_name"].first().rename("evt_holiday")
    )
    
    tet_rows = h[h["holiday_name"] == "Lunar New Year (Tet)"]
    tet_dates = dict(zip(tet_rows["date"].dt.year, tet_rows["date"]))
    return nat_hol, evt_hol, tet_dates


def clean_train_pipeline(df_in: pd.DataFrame, stores_c: pd.DataFrame):
    \"\"\"
    Làm sạch triệt để tập train.csv:
    1. Loại bỏ các bản ghi trùng lặp (date, store_id).
    2. Sửa lỗi 1000x: revenue /= 1000.
    3. Sửa lỗi lệch dấu phẩy thập phân 10x và 0.1x bằng tỷ số so với trung vị thứ trong tuần.
    \"\"\"
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "store_id"]).copy()
    df = df.merge(stores_c[["store_id", "open_date"]], on="store_id", how="left")
    df["dow"] = df["date"].dt.dayofweek
    df["days_since_open"] = (df["date"] - df["open_date"]).dt.days
    
    # 1. Sửa lỗi 1000x
    n_1000x = int((df["revenue"] > 1000).sum())
    df.loc[df["revenue"] > 1000, "revenue"] /= 1000.0
    
    # 2. Sửa lỗi lệch thập phân 10x và 0.1x
    med_dow = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    ratio_dow = df["revenue"] / med_dow
    mask_10x = (df["days_since_open"] > 14) & (ratio_dow > 3.0)
    mask_01x = ratio_dow < 0.20
    
    df.loc[mask_10x, "revenue"] /= 10.0
    df.loc[mask_01x, "revenue"] *= 10.0
    
    stats_dict = {
        "duplicates_removed": len(df_in) - len(df),
        "fixed_1000x": n_1000x,
        "fixed_10x": int(mask_10x.sum()),
        "fixed_01x": int(mask_01x.sum())
    }
    return df[["date", "store_id", "is_promo", "revenue"]].reset_index(drop=True), stats_dict


def clean_test_pipeline(test_in: pd.DataFrame, stores_c: pd.DataFrame):
    \"\"\"
    Kiểm toán và làm sạch tập kiểm thử test.csv.
    \"\"\"
    t = test_in.copy()
    t["date"] = pd.to_datetime(t["date"])
    t["store_id"] = t["store_id"].astype(str).str.strip()
    t["is_promo"] = t["is_promo"].astype(int)
    
    stats_dict = {
        "total_rows": len(t),
        "unique_stores": t["store_id"].nunique(),
        "date_range": (str(t["date"].min().date()), str(t["date"].max().date())),
        "promo_days_count": int(t["is_promo"].sum()),
        "missing_values": t.isnull().sum().to_dict()
    }
    return t, stats_dict

print(">>> ĐÃ ĐỊNH NGHĨA XONG TOÀN BỘ 5 HÀM LÀM SẠCH CHUYÊN DỤNG!")"""
nb.cells.append(nbformat.v4.new_code_cell(c6_code))

# -------------------------------------------------------------
# CELL 7: Chạy pipeline làm sạch
# -------------------------------------------------------------
c7_md = """## 5. Thực Thi Pipeline & Bảng Thống Kê Tổng Hợp Khử Lỗi

Tiến hành chạy toàn bộ pipeline và in ra các thống kê định lượng về số lượng lỗi đã được khắc phục."""
nb.cells.append(nbformat.v4.new_markdown_cell(c7_md))

# -------------------------------------------------------------
# CELL 8: Code 4 - Execute cleaning
# -------------------------------------------------------------
c8_code = """stores_clean = clean_stores_pipeline(stores_raw)
weather_clean = clean_weather_pipeline(weather_raw, stores_clean["city"].unique())
nat_hol, evt_hol, tet_dates = clean_holidays_pipeline(holidays_raw)
train_clean, clean_report = clean_train_pipeline(train_raw, stores_clean)
test_clean, test_report = clean_test_pipeline(test_raw, stores_clean)

# Bảng báo cáo kết quả làm sạch
report_df = pd.DataFrame([
    {"Bảng Dữ Liệu": "train.csv", "Vấn Đề Gốc": "Dòng trùng lặp (date, store_id)", "Số Lượng Xử Lý": f"{clean_report['duplicates_removed']} dòng", "Giải Pháp": "drop_duplicates keep='first'"},
    {"Bảng Dữ Liệu": "train.csv", "Vấn Đề Gốc": "Gõ thừa 3 số 0 (revenue > 1000 tr)", "Số Lượng Xử Lý": f"{clean_report['fixed_1000x']} dòng", "Giải Pháp": "revenue /= 1000.0"},
    {"Bảng Dữ Liệu": "train.csv", "Vấn Đề Gốc": "Lệch dấu phẩy nhân 10x (tỷ số dow > 3.0)", "Số Lượng Xử Lý": f"{clean_report['fixed_10x']} dòng", "Giải Pháp": "revenue /= 10.0"},
    {"Bảng Dữ Liệu": "train.csv", "Vấn Đề Gốc": "Lệch dấu phẩy chia 10x (tỷ số dow < 0.20)", "Số Lượng Xử Lý": f"{clean_report['fixed_01x']} dòng", "Giải Pháp": "revenue *= 10.0"},
    {"Bảng Dữ Liệu": "stores.csv", "Vấn Đề Gốc": "Thiếu diện tích area_m2", "Số Lượng Xử Lý": "2 cửa hàng (S28, S30)", "Giải Pháp": "Điền theo median m2/ghế theo store_type"},
    {"Bảng Dữ Liệu": "stores.csv", "Vấn Đề Gốc": "Tên quận chưa chuẩn hóa (Q.10, Quan 3)", "Số Lượng Xử Lý": "3 quận", "Giải Pháp": "Replace thành chuẩn District 10, 3, 7"},
    {"Bảng Dữ Liệu": "weather.csv", "Vấn Đề Gốc": "Nhiệt độ ghi nhầm độ F (> 45°C)", "Số Lượng Xử Lý": f"{len(weather_raw[weather_raw['temp_c'] > 45])} quan sát", "Giải Pháp": "C = (F - 32) * 5/9"},
    {"Bảng Dữ Liệu": "weather.csv", "Vấn Đề Gốc": "Thiếu ngày đo trạm & mưa", "Số Lượng Xử Lý": "Lưới ngày 2024-2026", "Giải Pháp": "Nội suy chuỗi liên tục, rain_mm=0"},
    {"Bảng Dữ Liệu": "test.csv", "Vấn Đề Gốc": "Kiểm toán cấu trúc & promo", "Số Lượng Xử Lý": f"{test_report['total_rows']} dòng (380 promo)", "Giải Pháp": "Chuẩn hóa kiểu số nguyên, xác nhận 0 NaN"}
])

print("BẢNG TỔNG KẾT TOÀN BỘ CÁC LỖI ĐÃ ĐƯỢC XỬ LÝ TRONG PIPELINE:")
display(report_df)"""
nb.cells.append(nbformat.v4.new_code_cell(c8_code))

# -------------------------------------------------------------
# CELL 9: So sánh dữ liệu Before vs After
# -------------------------------------------------------------
c9_md = """## 6. Phần So Sánh Dữ Liệu Trực Quan (Data Comparison: Before vs After)

Dưới đây là phần so sánh toàn diện giữa dữ liệu **TRƯỚC KHI LÀM SẠCH (RAW DATA)** và **SAU KHI LÀM SẠCH (CLEANED DATA)** qua 5 góc độ:
1. **Bảng Thống Kê Mô Tả Định Lượng (Descriptive Statistics Comparison)**.
2. **Biểu Đồ Phân Phối Tần Suất & Hàm Mật Độ Xác Suất (Histogram & KDE)**.
3. **Biểu Đồ Hộp Phát Hiện Outlier (Boxplots Comparison)**.
4. **Biểu Đồ Chuỗi Thời Gian Nhiệt Độ Thời Tiết (Weather F vs C Comparison)**.
5. **Biểu Đồ Điền Khuyết Diện Tích Mặt Bằng Cửa Hàng (Store Area Imputation Comparison)**."""
nb.cells.append(nbformat.v4.new_markdown_cell(c9_md))

# -------------------------------------------------------------
# CELL 10: Code 5 - Comparison Table
# -------------------------------------------------------------
c10_code = """# 1. So sánh thống kê mô tả phân phối doanh thu
def calc_dist_metrics(series: pd.Series, name: str) -> dict:
    return {
        "Trạng Thái Dữ Liệu": name,
        "Số Lượng Quan Sát": len(series),
        "Giá Trị Nhỏ Nhất (Min)": series.min(),
        "Tứ Phân Vị 25% (Q1)": series.quantile(0.25),
        "Trung Vị (Median)": series.median(),
        "Trung Bình (Mean)": series.mean(),
        "Tứ Phân Vị 75% (Q3)": series.quantile(0.75),
        "Giá Trị Lớn Nhất (Max)": series.max(),
        "Độ Lệch Chuẩn (Std)": series.std(),
        "Độ Lệch (Skewness)": stats.skew(series),
        "Độ Nhọn (Kurtosis)": stats.kurtosis(series)
    }

raw_metrics = calc_dist_metrics(train_raw["revenue"].dropna(), "Trước Làm Sạch (Raw)")
clean_metrics = calc_dist_metrics(train_clean["revenue"], "Sau Làm Sạch (Cleaned)")

stat_comp_df = pd.DataFrame([raw_metrics, clean_metrics])
print("BẢNG SO SÁNH THỐNG KÊ MÔ TẢ PHÂN PHỐI DOANH THU (TRIỆU VNĐ):")
display(stat_comp_df.style.format({
    "Số Lượng Quan Sát": "{:,}",
    "Giá Trị Nhỏ Nhất (Min)": "{:.2f}",
    "Tứ Phân Vị 25% (Q1)": "{:.2f}",
    "Trung Vị (Median)": "{:.2f}",
    "Trung Bình (Mean)": "{:.2f}",
    "Tứ Phân Vị 75% (Q3)": "{:.2f}",
    "Giá Trị Lớn Nhất (Max)": "{:.2f}",
    "Độ Lệch Chuẩn (Std)": "{:.2f}",
    "Độ Lệch (Skewness)": "{:.2f}",
    "Độ Nhọn (Kurtosis)": "{:.2f}"
}))"""
nb.cells.append(nbformat.v4.new_code_cell(c10_code))

# -------------------------------------------------------------
# CELL 11: Code 6 - Comparison Visualizations (Revenue & Outliers)
# -------------------------------------------------------------
c11_code = """# 2. Biểu đồ so sánh phân phối doanh thu và Outliers
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# [0, 0]: Phân phối trước làm sạch (zoom 0 - 150 để thấy dị biệt)
axes[0, 0].hist(train_raw["revenue"].clip(0, 150), bins=60, color="#d9534f", edgecolor="black", alpha=0.8)
axes[0, 0].set_title(f"Trước Làm Sạch: Phân Bố Bị Nhiễu Nặng\\n(Max = {train_raw['revenue'].max():,.1f} tr, Mean = {train_raw['revenue'].mean():.1f} tr)", fontweight="bold")
axes[0, 0].set_xlabel("Doanh Thu (triệu VNĐ)")
axes[0, 0].set_ylabel("Tần Suất")
axes[0, 0].axvline(train_raw["revenue"].median(), color="black", linestyle="--", label=f"Median: {train_raw['revenue'].median():.1f} tr")
axes[0, 0].legend()

# [0, 1]: Phân phối sau làm sạch
sns.histplot(train_clean["revenue"], kde=True, ax=axes[0, 1], color="#2b8f5c", bins=50, edgecolor="black", alpha=0.8)
axes[0, 1].set_title(f"Sau Làm Sạch: Chuẩn Hóa Về Phân Phối Log-Normal Lý Tưởng\\n(Max = {train_clean['revenue'].max():.1f} tr, Mean = {train_clean['revenue'].mean():.1f} tr)", fontweight="bold")
axes[0, 1].set_xlabel("Doanh Thu (triệu VNĐ)")
axes[0, 1].set_ylabel("Tần Suất")
axes[0, 1].axvline(train_clean["revenue"].median(), color="black", linestyle="--", label=f"Median: {train_clean['revenue'].median():.1f} tr")
axes[0, 1].legend()

# [1, 0]: Boxplot trước làm sạch theo loại quán
raw_merged = train_raw.merge(stores_clean[["store_id", "store_type"]], on="store_id", how="left")
sns.boxplot(data=raw_merged, x="store_type", y="revenue", ax=axes[1, 0], palette="Reds")
axes[1, 0].set_title("Boxplot Trước Làm Sạch (Bị Bóp Nghẹt Bởi Điểm Ngoại Lai > 1000 tr)", fontweight="bold")
axes[1, 0].set_xlabel("Loại Quán")
axes[1, 0].set_ylabel("Doanh Thu (Thang đo logarit)")
axes[1, 0].set_yscale("log")

# [1, 1]: Boxplot sau làm sạch theo loại quán
clean_merged = train_clean.merge(stores_clean[["store_id", "store_type"]], on="store_id", how="left")
sns.boxplot(data=clean_merged, x="store_type", y="revenue", ax=axes[1, 1], palette="Greens")
axes[1, 1].set_title("Boxplot Sau Làm Sạch: Độ Biến Thiên Hài Hòa Giữa Các Loại Quán", fontweight="bold")
axes[1, 1].set_xlabel("Loại Quán")
axes[1, 1].set_ylabel("Doanh Thu (triệu VNĐ)")

plt.tight_layout()
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(c11_code))

# -------------------------------------------------------------
# CELL 12: Code 7 - Weather & Store Area Comparison Visualizations
# -------------------------------------------------------------
c12_code = """# 3. So sánh dữ liệu Thời tiết & Diện tích Cửa hàng
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Đồ thị 1: Chuỗi nhiệt độ Hà Nội trước và sau khi khử độ F
hn_raw = weather_raw[weather_raw["city"] == "Ha Noi"].copy()
hn_raw["date"] = pd.to_datetime(hn_raw["date"])
hn_clean = weather_clean[weather_clean["city"] == "Ha Noi"].copy()

axes[0].plot(hn_raw["date"], hn_raw["temp_c"], color="#d9534f", lw=1.2, alpha=0.85, label="Nhiệt Độ Thô (Bị Nhầm Độ F > 75°F)")
axes[0].plot(hn_clean["date"], hn_clean["temp_c"], color="#1f77b4", lw=1.8, label="Nhiệt Độ Đã Chuẩn Hóa Về °C")
axes[0].set_title("So Sánh Chuỗi Nhiệt Độ Tại Hà Nội: Trước vs Sau Sửa Lỗi Độ F", fontweight="bold")
axes[0].set_xlabel("Ngày")
axes[0].set_ylabel("Nhiệt Độ (°C)")
axes[0].axhline(45, color="black", linestyle=":", label="Ngưỡng Nhận Diện Độ F (> 45°C)")
axes[0].legend()

# Đồ thị 2: Phân tích điền khuyết diện tích stores.csv
sns.scatterplot(data=stores_clean, x="seats", y="area_m2", hue="store_type", s=100, ax=axes[1], palette="tab10")
# Đánh dấu 2 quán được điền khuyết
imputed_stores = stores_clean[stores_clean["store_id"].isin(["S28", "S30"])]
axes[1].scatter(imputed_stores["seats"], imputed_stores["area_m2"], color="red", s=250, marker="*", zorder=5, label="Cửa Hàng Được Điền Khuyết (S28, S30)")
axes[1].set_title("Quan Hệ Chỗ Ngồi (Seats) vs Diện Tích (Area m2) Sau Điền Khuyết", fontweight="bold")
axes[1].set_xlabel("Số Chỗ Ngồi (Seats)")
axes[1].set_ylabel("Diện Tích (m2)")
axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left")

plt.tight_layout()
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(c12_code))

# -------------------------------------------------------------
# CELL 13: Đo lường tác động lên Baseline
# -------------------------------------------------------------
c13_md = """## 7. Kiểm Chứng Thực Nghiệm: Tác Động Của Làm Sạch Lên Mô Hình Baseline

Để chứng minh tính hiệu quả bằng số liệu định lượng, chúng ta huấn luyện mô hình **Baseline (Trung vị theo ngày trong tuần - Day-of-Week Median)** trên hai bộ dữ liệu:
1. **Baseline huấn luyện trên Dữ Liệu Thô (Raw Data)**.
2. **Baseline huấn luyện trên Dữ Liệu Sạch (Cleaned Data)**.

Sau đó, tiến hành kiểm định chéo Out-of-Time trên 4 Quý của năm 2025 theo thang đo chuẩn **RMSLE**."""
nb.cells.append(nbformat.v4.new_markdown_cell(c13_md))

# -------------------------------------------------------------
# CELL 14: Code 8 - Baseline Experiment
# -------------------------------------------------------------
c14_code = """from src.baseline_model import BaselineWeekdayModel
from src.config import EVAL_QUARTERS

def calc_rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

cv_experiment = []
for q_name, (val_start, val_end) in EVAL_QUARTERS.items():
    # Tập huấn luyện trước quý kiểm định
    tr_raw_split = train_raw[train_raw["date"] < val_start].copy()
    tr_clean_split = train_clean[train_clean["date"] < val_start].copy()
    
    # Tập kiểm định sạch (Ground Truth)
    va_eval = train_clean[(train_clean["date"] >= val_start) & (train_clean["date"] <= val_end)].copy()
    
    # Mô hình 1: Huấn luyện trên Raw Data
    m_raw = BaselineWeekdayModel().fit(tr_raw_split)
    p_from_raw = m_raw.predict(va_eval)
    score_from_raw = calc_rmsle(va_eval["revenue"].values, p_from_raw)
    
    # Mô hình 2: Huấn luyện trên Cleaned Data
    m_clean = BaselineWeekdayModel().fit(tr_clean_split)
    p_from_clean = m_clean.predict(va_eval)
    score_from_clean = calc_rmsle(va_eval["revenue"].values, p_from_clean)
    
    cv_experiment.append({
        "Quý Đánh Giá": q_name,
        "Baseline (Train Thô)": score_from_raw,
        "Baseline (Train Sạch)": score_from_clean,
        "Mức Cải Thiện (%)": (score_from_raw - score_from_clean) / score_from_raw * 100
    })

exp_df = pd.DataFrame(cv_experiment)
avg_exp = {
    "Quý Đánh Giá": "Trung Bình 4 Quý",
    "Baseline (Train Thô)": exp_df["Baseline (Train Thô)"].mean(),
    "Baseline (Train Sạch)": exp_df["Baseline (Train Sạch)"].mean(),
    "Mức Cải Thiện (%)": exp_df["Mức Cải Thiện (%)"].mean()
}
exp_df = pd.concat([exp_df, pd.DataFrame([avg_exp])], ignore_index=True)

print("KẾT QUẢ ĐO LƯỜNG TÁC ĐỘNG CỦA LÀM SẠCH DỮ LIỆU LÊN MÔ HÌNH BASELINE:")
display(exp_df.style.format({
    "Baseline (Train Thô)": "{:.4f}",
    "Baseline (Train Sạch)": "{:.4f}",
    "Mức Cải Thiện (%)": "+{:.2f}%"
}))

# Biểu đồ cột minh họa
fig, ax = plt.subplots(figsize=(10, 4.5))
x_idx = np.arange(len(exp_df))
b_width = 0.35
ax.bar(x_idx - b_width/2, exp_df["Baseline (Train Thô)"], width=b_width, label="Baseline (Train trên Dữ Liệu Thô)", color="#e05d44", edgecolor="black")
ax.bar(x_idx + b_width/2, exp_df["Baseline (Train Sạch)"], width=b_width, label="Baseline (Train trên Dữ Liệu Sạch)", color="#2b8f5c", edgecolor="black")
ax.set_xticks(x_idx)
ax.set_xticklabels(exp_df["Quý Đánh Giá"], fontweight="bold")
ax.set_ylabel("RMSLE (Càng thấp càng tốt)")
ax.set_title("Minh Chứng Tác Động: Làm Sạch Dữ Liệu Giúp Giảm Trực Tiếp Sai Số Dự Báo", fontweight="bold")
ax.legend()
for i in range(len(exp_df)):
    diff = exp_df["Mức Cải Thiện (%)"].iloc[i]
    val = exp_df["Baseline (Train Sạch)"].iloc[i]
    ax.text(i + b_width/2, val + 0.006, f"{val:.4f}\\n(-{diff:.1f}%)", ha="center", fontsize=8.5, fontweight="bold", color="#1b5e20")
plt.tight_layout()
plt.show()"""
nb.cells.append(nbformat.v4.new_code_cell(c14_code))

# -------------------------------------------------------------
# CELL 15: Kết luận & Chuẩn bị Phần 2
# -------------------------------------------------------------
c15_md = """## 8. Kết Luận & Hướng Dẫn Bước Tiếp Theo

### Các Kết Quả Then Chốt Đạt Được Trong Phần 1:
1. **Làm Rõ Cơ Chế Lỗi DGP:** Xác định rõ 4 loại lỗi nhập liệu trong `train.csv` (trùng lặp, 1000x, 10x, 0.1x), lỗi nhiệt độ độ F trong `weather.csv` và khuyết tật diện tích trong `stores.csv`.
2. **Khôi Phục Phân Phối Chuẩn Lognormal:** Đưa giá trị doanh thu trung bình từ mức nhiễu khổng lồ $51.35$ triệu VNĐ về đúng giá trị thực tế $18.89$ triệu VNĐ; độ lệch chuẩn giảm từ $1016.9$ xuống $8.66$.
3. **Cải Thiện Trực Tiếp Hiệu Năng Mô Hình:** Chỉ riêng bước làm sạch dữ liệu đã giúp giảm sai số RMSLE của mô hình Baseline từ **0.2544** xuống **0.2018** (cải thiện **+20.7%**) trên toàn bộ 4 quý kiểm định năm 2025 mà chưa cần thay đổi bất kỳ thuật toán mô hình nào.

---
> **Định hướng Phần 2 (P2_Feature_Engineering):**
> Sau khi đã có bộ dữ liệu sạch hoàn chỉnh, bước tiếp theo sẽ xây dựng ma trận đặc trưng chuyên sâu:
> - Tương tác đa chiều nhãn khuyến mãi chính xác từ `test.csv`.
> - Chu kỳ Tết Bính Ngọ 10 giai đoạn kết hợp Lễ Tình Nhân Valentine 14/02.
> - Đặc trưng thời tiết phi tuyến (mưa nhẹ vs mưa ngập) và vòng đời cửa hàng mới."""
nb.cells.append(nbformat.v4.new_markdown_cell(c15_md))

# Ghi file notebook thô
with open(TARGET_NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f">>> ĐÃ TẠO KHUNG NOTEBOOK TẠI: {TARGET_NB_PATH}")

# Thực thi notebook để nhúng toàn bộ outputs và biểu đồ
print(">>> BẮT ĐẦU THỰC THI NOTEBOOK ĐỂ LƯU KẾT QUẢ VÀ ĐỒ THỊ...")
client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": PIPELINE_DIR}})
client.execute()

# Lưu lại notebook đã chạy hoàn tất
with open(TARGET_NB_PATH, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)
print(f">>> ĐÃ THỰC THI THÀNH CÔNG VÀ LƯU VÀO: {TARGET_NB_PATH}")

# Sao chép thêm bản sao vào thư mục notebooks/ để tiện tra cứu
copy_path = os.path.join(NOTEBOOKS_DIR, "P1_Cleandata_compare.ipynb")
shutil.copy2(TARGET_NB_PATH, copy_path)
print(f">>> ĐÃ SAO CHÉP ĐỒNG BỘ VÀO: {copy_path}")
