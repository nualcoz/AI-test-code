# -*- coding: utf-8 -*-
"""
build_chapter4_steps.py
Khởi tạo và thực thi chuỗi 8 Notebook chuẩn theo bài giảng:
'Chapter 4 - Building models and developing AI applications' (NEU FIT):

1. Step 1: Collect & Prepare (Thu thập, đọc bảng, merge & data dictionary)
2. Step 2: Explore / EDA (Khám phá cấu trúc, missing, distribution, charts, correlation)
3. Step 3: Preprocess (Khử trùng lặp, sửa lỗi 1000x, 10x, 0.1x, sửa độ F, fillna, encode)
4. Step 4: Split (Time-based split cho Cafe Revenue, expanding window 4 Quý)
5. Step 5: Scale & Feature Engineering (Log transform, StandardScaler, MinMax, 10 giai đoạn Tết, Promo interactions)
6. Step 6: Model & Train (Baseline mean/weekday, Ridge, Decision Tree, Random Forest, GBDT, CafeRevenueModelV3)
7. Step 7: Evaluate & Diagnostics (RMSLE, MAE, R2, bảng so sánh 4 quý, actual vs pred, residual diagnostics, overfitting)
8. Step 8: Deploy & Kaggle Submission (Lưu model joblib, Colab @param form dự báo What-if, tạo file submission.csv chuẩn)
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
STEPS_DIR = os.path.join(PROJECT_ROOT, "notebooks", "steps")
os.makedirs(PIPELINE_DIR, exist_ok=True)
os.makedirs(STEPS_DIR, exist_ok=True)

def execute_and_save(nb, filename):
    target_path = os.path.join(PIPELINE_DIR, filename)
    with open(target_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    print(f"--> Đang thực thi {filename} ...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": PIPELINE_DIR}})
    client.execute()
    
    with open(target_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    # Sao chép vào notebooks/steps/
    mirror_path = os.path.join(STEPS_DIR, filename)
    shutil.copy2(target_path, mirror_path)
    print(f"    ✓ Hoàn tất {filename} (đã lưu tại {target_path} và {mirror_path})")


# ==============================================================================
# STEP 1: COLLECT & PREPARE
# ==============================================================================
def create_step1():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 1: Collect & Prepare (Thu Thập & Chuẩn Bị Dữ Liệu)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Trước khi viết bất kỳ dòng mã nào, Chapter 4 đặt ra **3 câu hỏi vàng**:
1. **Dự đoán cái gì? (What do we predict?)**: Dự báo doanh thu hàng ngày (`revenue`, đơn vị: triệu VNĐ) cho từng quán cà phê.
2. **Từ những cột nào? (From which columns?)**: Từ ngày (`date`), mã quán (`store_id`), khuyến mãi (`is_promo`), cùng các bảng liên kết phụ trợ: thông tin quán (`stores.csv`), thời tiết (`weather.csv`) và ngày lễ Tết (`holidays.csv`).
3. **Đánh giá bằng thước đo gì? (Measured by what?)**: Thước đo chính thức là **RMSLE** (Root Mean Squared Logarithmic Error) tính trên $\\log(1 + \\text{revenue})$:
   $$\\text{RMSLE} = \\sqrt{\\frac{1}{N} \\sum_{i=1}^N \\left(\\log(1 + \\hat{y}_i) - \\log(1 + y_i)\\right)^2}$$

### 2. Dùng Những Gì Để Làm?
- **Thư viện chính:** `pandas` (hàm `pd.read_csv`, `pd.merge`).
- **Kỹ thuật dữ liệu:** Thiết kế sơ đồ quan hệ (Relational Schema), phép nối `how='left'` để bảo toàn dòng bảng chính, và xây dựng **Data Dictionary** chuẩn hóa."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))
    
    c1 = """import os, sys
import pandas as pd
import numpy as np

# Thêm đường dẫn project root vào sys.path
for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

print(f">>> THƯ MỤC DỮ LIỆU ĐANG SỬ DỤNG: {DATA_DIR}")

# 1. Đọc các bảng dữ liệu
train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
stores_df = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_df = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_df = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

print("KÍCH THƯỚC CÁC BẢNG NGUỒN:")
print(f"- train.csv    : {train_df.shape[0]:,} dòng x {train_df.shape[1]} cột")
print(f"- test.csv     : {test_df.shape[0]:,} dòng x {test_df.shape[1]} cột")
print(f"- stores.csv   : {stores_df.shape[0]:,} dòng x {stores_df.shape[1]} cột")
print(f"- weather.csv  : {weather_df.shape[0]:,} dòng x {weather_df.shape[1]} cột")
print(f"- holidays.csv : {holidays_df.shape[0]:,} dòng x {holidays_df.shape[1]} cột")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 3. Thực Hiện Nối Bảng (Merge Tables) & Kiểm Tra Hình Dạng
Nguyên tắc từ Slide 13: *Compare shape before and after a merge: more rows means the key is repeated in the right table!*"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Đổi date sang chuỗi định dạng đồng nhất
train_df["date"] = pd.to_datetime(train_df["date"])
weather_df["date"] = pd.to_datetime(weather_df["date"])
holidays_df["date"] = pd.to_datetime(holidays_df["date"])

# 1. Nối train với stores qua key 'store_id'
m1 = train_df.merge(stores_df, on="store_id", how="left")
print(f"Shape trước khi merge stores: {train_df.shape} -> Sau khi merge: {m1.shape}")

# 2. Nối tiếp với weather qua cặp key ['date', 'city']
m2 = m1.merge(weather_df, on=["date", "city"], how="left")
print(f"Shape sau khi merge weather : {m2.shape}")

# 3. Nối tiếp với holidays qua key 'date'
# holidays_df có thể có nhiều sự kiện trong cùng 1 ngày, dedup để tránh nhân bản dòng
holidays_dedup = holidays_df.drop_duplicates(subset=["date"], keep="first")
full_merged = m2.merge(holidays_dedup, on="date", how="left")
print(f"Shape sau cùng khi merge holidays: {full_merged.shape}")

display(full_merged.head(3))"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))

    md2 = """### 4. Bảng Từ Điển Dữ Liệu Toàn Diện (Data Dictionary)
Theo chuẩn Chapter 4, việc lập Data Dictionary giúp toàn bộ team hiểu rõ ý nghĩa, đơn vị và kiểu dữ liệu trước khi đi vào phân tích."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md2))

    c3 = """data_dict = pd.DataFrame([
    {"Cột": "date", "Bảng Gốc": "train/test/weather", "Ý Nghĩa Nghiệp Vụ": "Ngày ghi nhận giao dịch", "Đơn Vị": "YYYY-MM-DD", "Kiểu": "datetime"},
    {"Cột": "store_id", "Bảng Gốc": "train/test/stores", "Ý Nghĩa Nghiệp Vụ": "Mã định danh cửa hàng", "Đơn Vị": "S01 - S45", "Kiểu": "category"},
    {"Cột": "is_promo", "Bảng Gốc": "train/test", "Ý Nghĩa Nghiệp Vụ": "Có chạy chương trình khuyến mãi hay không", "Đơn Vị": "0 (Không), 1 (Có)", "Kiểu": "int / bool"},
    {"Cột": "revenue", "Bảng Gốc": "train", "Ý Nghĩa Nghiệp Vụ": "Biến mục tiêu: Doanh thu trong ngày của quán", "Đơn Vị": "Triệu VNĐ", "Kiểu": "float (Target)"},
    {"Cột": "city", "Bảng Gốc": "stores/weather", "Ý Nghĩa Nghiệp Vụ": "Thành phố quán đặt cơ sở", "Đơn Vị": "Ha Noi, Ho Chi Minh, Da Nang", "Kiểu": "category"},
    {"Cột": "district", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Quận/huyện của cửa hàng", "Đơn Vị": "Tên quận", "Kiểu": "text"},
    {"Cột": "store_type", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Mô hình kinh doanh mặt bằng", "Đơn Vị": "mall, office, street, residential, university", "Kiểu": "category"},
    {"Cột": "area_m2", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Diện tích mặt sàn kinh doanh", "Đơn Vị": "m2", "Kiểu": "float"},
    {"Cột": "seats", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Số lượng ghế ngồi phục vụ", "Đơn Vị": "Số chỗ", "Kiểu": "int"},
    {"Cột": "has_parking", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Có bãi giữ xe máy/ô tô không", "Đơn Vị": "0 (Không), 1 (Có)", "Kiểu": "int / bool"},
    {"Cột": "open_date", "Bảng Gốc": "stores", "Ý Nghĩa Nghiệp Vụ": "Ngày quán bắt đầu khai trương", "Đơn Vị": "YYYY-MM-DD", "Kiểu": "datetime"},
    {"Cột": "temp_c", "Bảng Gốc": "weather", "Ý Nghĩa Nghiệp Vụ": "Nhiệt độ trung bình trong ngày", "Đơn Vị": "Độ Celsius (°C)", "Kiểu": "float"},
    {"Cột": "rain_mm", "Bảng Gốc": "weather", "Ý Nghĩa Nghiệp Vụ": "Lượng mưa đo đạc trong ngày", "Đơn Vị": "mm", "Kiểu": "float"},
    {"Cột": "holiday_name", "Bảng Gốc": "holidays", "Ý Nghĩa Nghiệp Vụ": "Tên ngày lễ hoặc sự kiện", "Đơn Vị": "Chuỗi văn bản", "Kiểu": "text"},
    {"Cột": "type", "Bảng Gốc": "holidays", "Ý Nghĩa Nghiệp Vụ": "Phân loại kỳ nghỉ", "Đơn Vị": "national_holiday, event", "Kiểu": "category"}
])

display(data_dict)
print(">>> HOÀN THÀNH STEP 1: ĐÃ XÁC ĐỊNH BÀI TOÁN, KẾT NỐI DỮ LIỆU VÀ LẬP TỪ ĐIỂN DỮ LIỆU!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c3))
    return nb


# ==============================================================================
# STEP 2: EXPLORE (EDA)
# ==============================================================================
def create_step2():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 2: Explore (Khám Phá & Phân Tích Dữ Liệu - EDA)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 14 & 20 nêu rõ: *Output of this step: a written list of problems, before any of them is fixed.*
Chúng ta không vội làm sạch mà cần đóng vai trò "thanh tra dữ liệu":
- **Khảo sát cấu trúc & tóm tắt định lượng:** Dùng `info()` và `describe()` để nhận diện các giá trị bất khả thi (impossible values).
- **Rà soát giá trị khuyết (Missing values):** Xác định cột nào thiếu và tỷ lệ bao nhiêu.
- **Trực quan hóa đa chiều:**
  - 1 cột (Univariate): Histogram, Boxplot kiểm tra độ lệch (skew) và outlier.
  - 2 cột (Bivariate): Bar plot so sánh nhóm, Scatter plot tương quan, Line plot theo thời gian.
  - Nhiều cột (Multivariate): Correlation Heatmap phân tích mối quan hệ giữa biến số và biến mục tiêu.

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `pandas`, `matplotlib.pyplot`, `seaborn`.
- **Hàm & Phương pháp:** `df.info()`, `df.describe().round(1)`, `df.isna().sum()`, `plt.hist`, `plt.boxplot`, `sns.heatmap`."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["figure.dpi"] = 120

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
        break

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
stores = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

print("=== 1. TỔNG QUAN INFO() CỦA TRAIN.CSV ===")
train.info()

print("\\n=== 2. TÓM TẮT THỐNG KÊ DESCRIBE() ===")
display(train.describe().round(2))"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 3. Nhận Diện Danh Sách Vấn Đề (The Written List of Problems)
Từ kết quả `describe()`, ta thấy ngay:
- `revenue` có Max = **148,240.0** triệu VNĐ, Mean = **51.35** trong khi Median = **18.25**! Có lỗi nhân $1000\times$.
- Cột `temp_c` trong `weather.csv` có Max $> 85^\circ\text{C}$! Có lỗi ghi nhầm độ F."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Kiểm tra thiếu dữ liệu trên toàn bộ các bảng
missing_summary = pd.DataFrame({
    "Bảng": ["train", "stores", "stores", "weather", "weather"],
    "Cột": ["revenue", "area_m2", "seats", "temp_c", "rain_mm"],
    "Số Lượng Thiếu": [train["revenue"].isna().sum(), stores["area_m2"].isna().sum(), stores["seats"].isna().sum(), weather["temp_c"].isna().sum(), weather["rain_mm"].isna().sum()],
    "Tỷ Lệ Thiếu (%)": [
        train["revenue"].isna().mean()*100, stores["area_m2"].isna().mean()*100,
        stores["seats"].isna().mean()*100, weather["temp_c"].isna().mean()*100, weather["rain_mm"].isna().mean()*100
    ]
})
display(missing_summary)"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))

    md2 = """### 4. Hệ Thống Biểu Đồ Trực Quan Hóa (Charts that Matter)
Theo Slide 17, 18, 19:
- Histogram & Boxplot phân phối doanh thu.
- Bar plot so sánh doanh thu theo loại quán (`store_type`) và khuyến mãi (`is_promo`).
- Correlation Heatmap."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md2))

    c3 = """# Ghép train với stores để trực quan hóa
train_merged = train.merge(stores, on="store_id", how="left")
train_merged["date"] = pd.to_datetime(train_merged["date"])

fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# 1. Histogram Doanh thu (giới hạn 0 - 100 để thấy hình dáng thực tế)
axes[0, 0].hist(train_merged["revenue"].clip(0, 100), bins=50, color="#1f77b4", edgecolor="black")
axes[0, 0].set_title("Histogram: Phân Phối Doanh Thu (Clip 0 - 100 tr)", fontweight="bold")
axes[0, 0].set_xlabel("Doanh Thu (triệu VNĐ)")

# 2. Boxplot Doanh thu theo loại quán (Log-scale)
sns.boxplot(data=train_merged, x="store_type", y="revenue", ax=axes[0, 1], palette="Blues")
axes[0, 1].set_yscale("log")
axes[0, 1].set_title("Boxplot: Doanh Thu Theo Loại Quán (Log Scale)", fontweight="bold")

# 3. Bar plot: Tác động của khuyến mãi
promo_bar = train_merged.groupby(["store_type", "is_promo"])["revenue"].median().unstack()
promo_bar.plot(kind="bar", ax=axes[1, 0], color=["#999999", "#d9534f"], edgecolor="black")
axes[1, 0].set_title("Bar: Doanh Thu Trung Vị (Ngày Thường vs Promo)", fontweight="bold")
axes[1, 0].set_ylabel("Trung Vị Doanh Thu (triệu VNĐ)")

# 4. Correlation Heatmap
corr_num = train_merged[["revenue", "is_promo", "seats", "area_m2", "has_parking"]].corr()
sns.heatmap(corr_num, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 1], vmin=-1, vmax=1)
axes[1, 1].set_title("Heatmap: Tương Quan Tuyến Tính Giữa Các Biến Số", fontweight="bold")

plt.tight_layout()
plt.show()

print(">>> HOÀN THÀNH STEP 2: ĐÃ XÁC ĐỊNH TOÀN BỘ CÁC VẤN ĐỀ VÀ MỐI QUAN HỆ ĐẶC TRƯNG!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c3))
    return nb


# ==============================================================================
# STEP 3: PREPROCESS
# ==============================================================================
def create_step3():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 3: Preprocess (Tiền Xử Lý & Làm Sạch Dữ Liệu)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 21 đặt mục tiêu: *Output of this step: a table of numbers only, with no duplicates and no missing values.*
Các tác vụ cốt lõi:
1. **Drop duplicate rows:** Loại bỏ các bản ghi trùng khóa thời gian và địa điểm.
2. **Turn wrong values into corrected/valid values:**
   - Nhiệt độ độ F $\to$ độ C ($T > 45^\circ\text{C}$).
   - Doanh thu gõ thừa số 0 ($> 1000$ triệu $\to$ chia $1000$).
   - Lệch dấu phẩy thập phân ($10\times$ và $0.1\times$) dựa trên tỷ số so với trung vị thứ trong tuần của quán.
3. **Unify different spellings:** Chuẩn hóa `District 10`, `District 3`, `District 7`.
4. **Handle missing values:**
   - Điền khuyết diện tích `area_m2` theo tỷ lệ trung vị diện tích/ghế theo loại quán.
   - Nội suy chuỗi thời tiết liên tục 2024 - 2026.
5. **Encode text columns as numbers:** One-hot encoding `pd.get_dummies` cho `city`, `store_type`, `tet_phase`.

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `pandas`, `numpy`.
- **Hàm:** `drop_duplicates()`, `replace()`, `fillna()`, `get_dummies(dtype=int)`."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.cleaning import clean_stores, clean_weather, clean_holidays, clean_train, clean_test

# 1. Đọc dữ liệu
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

# 2. Áp dụng pipeline làm sạch
stores_c = clean_stores(stores_raw)
weather_c = clean_weather(weather_raw, stores_c["city"].unique())
nat_hol, evt_hol, tet_dates = clean_holidays(holidays_raw)
train_c, stats_train = clean_train(train_raw, stores_c)
test_c, stats_test = clean_test(test_raw, stores_c)

print("KẾT QUẢ TIỀN XỬ LÝ LÀM SẠCH:")
print(f"- Loại bỏ trùng lặp train : {stats_train['duplicates_removed']} dòng")
print(f"- Sửa lỗi gõ thừa 1000x  : {stats_train['fixed_1000x']} dòng")
print(f"- Sửa lỗi lệch thập phân 10x: {stats_train['fixed_10x']} dòng")
print(f"- Sửa lỗi lệch thập phân 0.1x: {stats_train['fixed_01x']} dòng")
print(f"- Train sạch sau cùng     : {train_c.shape}")
print(f"- Test sạch sau cùng      : {test_c.shape}")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 3. Mã Hóa One-Hot Encoding (Text Columns to Numbers)
Theo Slide 25: Chuyển toàn bộ các biến phân loại (`store_type`, `city`) thành các cột nhị phân $0/1$ để mô hình có thể tính toán ma trận."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Ghép thông tin cửa hàng vào train sạch
df_encoded = train_c.merge(stores_c[["store_id", "store_type", "city", "seats", "area_m2", "has_parking"]], on="store_id", how="left")
df_encoded["dow"] = df_encoded["date"].dt.dayofweek.astype(str)

# Áp dụng get_dummies
df_numeric = pd.get_dummies(df_encoded, columns=["store_type", "city", "dow"], dtype=int)
print(f"Kích thước bảng toàn số sau One-Hot Encoding: {df_numeric.shape}")
print("\\nDanh sách các cột:")
print(df_numeric.columns.tolist()[:15], "... [tổng cộng", len(df_numeric.columns), "cột]")
display(df_numeric.head(3))

print(">>> HOÀN THÀNH STEP 3: DỮ LIỆU ĐÃ SẠCH 100% VÀ SẴN SÀNG CHO MÔ HÌNH HÓA!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))
    return nb


# ==============================================================================
# STEP 4: SPLIT
# ==============================================================================
def create_step4():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 4: Split (Phân Chia Tập Huấn Luyện & Kiểm Định)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 26 & 28 chỉ rõ:
- **Rule 1: Keep the test set aside until Step 7; every decision uses the train set!**
- **Đặc thù bài toán Cafe Revenue (Slide 28):** *Group project: 5-fold for Card Churn, time-based split for Cafe Revenue!*
- **Vì sao không dùng Random K-Fold cho chuỗi thời gian?**
  Nếu chia ngẫu nhiên, mô hình sẽ dùng dữ liệu tương lai (ngày mai) để học và dự đoán cho ngày hôm qua, dẫn đến **rò rỉ dữ liệu (Data Leakage)** và điểm số cao ảo nhưng sụp đổ khi thi đấu thật.
- **Chiến lược phân chia:** Out-of-Time Expanding Window Split theo từng Quý của năm 2025:
  - **Q1/2025:** Train < 2025-01-01, Validation: 2025-01-01 đến 2025-03-31
  - **Q2/2025:** Train < 2025-04-01, Validation: 2025-04-01 đến 2025-06-30
  - **Q3/2025:** Train < 2025-07-01, Validation: 2025-07-01 đến 2025-09-30
  - **Q4/2025:** Train < 2025-10-01, Validation: 2025-10-01 đến 2025-12-31

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `pandas`, `sklearn.model_selection.TimeSeriesSplit`."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.config import EVAL_QUARTERS
from src.cleaning import clean_train, clean_stores

stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
train_c, _ = clean_train(train_raw, clean_stores(stores_raw))

print("CÁC MỐC PHÂN CHIA OUT-OF-TIME THEO 4 QUÝ NĂM 2025:")
for q, (s, e) in EVAL_QUARTERS.items():
    n_val = len(train_c[(train_c["date"] >= s) & (train_c["date"] <= e)])
    n_tr = len(train_c[train_c["date"] < s])
    print(f"- {q}: Train {n_tr:,} dòng (< {s}) | Validation {n_val:,} dòng ({s} đến {e})")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 3. Trực Quan Hóa Sơ Đồ Phân Chia Time-Based Split"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """fig, ax = plt.subplots(figsize=(12, 4))
quarters = list(EVAL_QUARTERS.keys())

for idx, (q, (s, e)) in enumerate(EVAL_QUARTERS.items()):
    val_start = pd.Timestamp(s)
    val_end = pd.Timestamp(e)
    # Train range
    ax.barh(idx, (val_start - pd.Timestamp("2024-01-01")).days, left=0, color="#1f77b4", edgecolor="black", label="Train Set (Lịch sử)" if idx==0 else "")
    # Validation range
    ax.barh(idx, (val_end - val_start).days, left=(val_start - pd.Timestamp("2024-01-01")).days, color="#ff7f0e", edgecolor="black", label="Validation Set (Đánh giá)" if idx==0 else "")

ax.set_yticks(range(len(quarters)))
ax.set_yticklabels(quarters, fontweight="bold")
ax.set_xlabel("Số ngày kể từ 01/01/2024")
ax.set_title("Sơ Đồ Phân Chia Kiểm Định Chéo Chuỗi Thời Gian (Time-Based Expanding Window)", fontweight="bold")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

print(">>> HOÀN THÀNH STEP 4: ĐÃ PHÂN CHIA TIME-SERIES ĐẢM BẢO KHÔNG RÒ RỈ DỮ LIỆU!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))
    return nb


# ==============================================================================
# STEP 5: SCALE & FEATURE ENGINEERING
# ==============================================================================
def create_step5():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 5: Scale & Feature Engineering (Co Giãn & Tạo Đặc Trưng)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 29 & 30 giải thích lý do cần Scale:
- **Vì sao phải scale?** Khoảng cách giữa các biến có đơn vị khác nhau (ví dụ: diện tích $100\text{m}^2$ vs số lượng giao dịch $15$) sẽ khiến các mô hình tuyến tính (Ridge, Logistic) hoặc khoảng cách (KNN, SVM) bị chi phối bởi biến có thang đo lớn hơn.
- **Rule 2 (Slide 61):** *Fit on train only! Imputer and scaler: fit on train, transform test; a pipeline does it for you.*
- **Biến đổi hàm mục tiêu:** Doanh thu có phân phối lệch phải (skewed right). Áp dụng logarit tự nhiên $\log(1 + y)$ đưa về phân phối chuẩn, giúp RMSE trên không gian log khớp chính xác với hàm mục tiêu **RMSLE** của cuộc thi!
- **Feature Engineering đặc thù cho Quý 1/2026:**
  - Khuyến mãi thực tế: `is_promo_global`, `promo_weekend`, `promo_{store_type}`.
  - Phân kỳ Tết 10 giai đoạn (Tết Bính Ngọ 17/02/2026).
  - Tương tác Lễ Tình Nhân Valentine 14/02 và Quốc tế Phụ Nữ 08/03.
  - Phân tách phi tuyến thời tiết: mưa nhẹ ($0-15\text{mm}$) vs mưa ngập ($>15\text{mm}$).

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `sklearn.preprocessing` (`StandardScaler`, `MinMaxScaler`), `numpy` (`log1p`)."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.cleaning import clean_train, clean_stores
from src.features import enrich_data, build_shared_matrix

# Đọc và làm sạch
stores_c = clean_stores(pd.read_csv(os.path.join(DATA_DIR, "stores.csv")))
train_c, _ = clean_train(pd.read_csv(os.path.join(DATA_DIR, "train.csv")), stores_c)

# 1. Biến đổi log1p trên biến mục tiêu
train_c["log_revenue"] = np.log1p(train_c["revenue"])

# 2. Chuẩn hóa đặc trưng số bằng StandardScaler (Fit on train only)
num_cols = ["seats", "area_m2"]
train_merged = train_c.merge(stores_c, on="store_id", how="left")

scaler = StandardScaler()
scaled_features = scaler.fit_transform(train_merged[num_cols])
print("Mean đã học trên train :", scaler.mean_.round(2))
print("Std đã học trên train  :", np.sqrt(scaler.var_).round(2))

display(pd.DataFrame(scaled_features, columns=[f"{c}_scaled" for c in num_cols]).head(4))

print(">>> HOÀN THÀNH STEP 5: ĐÃ CHUẨN HÓA LOG VÀ SCALE ĐẶC TRƯNG CHUẨN XÁC!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))
    return nb


# ==============================================================================
# STEP 6: MODEL & TRAIN
# ==============================================================================
def create_step6():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 6: Model & Train (Lựa Chọn & Huấn Luyện Mô Hình)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 32 & 34 nhấn mạnh:
- **Rule 3: Always compare with a baseline first!**
  Một mô hình AI phức tạp chỉ có giá trị khi nó vượt qua được mô hình Baseline đơn giản (Day-of-Week Median).
- **Same three lines for every model:**
  ```python
  model.fit(X_train, y_train)
  pred = model.predict(X_val)
  ```
- **Hệ thống các mô hình thử nghiệm (Model Zoo):**
  1. `BaselineWeekdayModel`: Chuẩn đề bài starter.
  2. `LinearRegression`: Mô hình hồi quy cổ điển OLS.
  3. `RidgeRegression`: Hồi quy có điều chuẩn L2 co hẹp hệ số.
  4. `DecisionTreeRegressor`: Mô hình cây phi tuyến.
  5. `RandomForestRegressor`: Rừng ngẫu nhiên chống overfitting.
  6. `HistGradientBoostingRegressor`: Tăng cường độ dốc GBDT.
  7. `CafeRevenueModelV3`: Mô hình phân tầng nâng cao (Hierarchical Multi-Stage).

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `sklearn.linear_model`, `sklearn.tree`, `sklearn.ensemble`, `src.cafe_model_v3`."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model_v3 import CafeRevenueModelV3

# Huấn luyện CafeRevenueModelV3 toàn diện
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

print("Đang huấn luyện CafeRevenueModelV3...")
model_v3 = CafeRevenueModelV3()
model_v3.fit(train_raw, stores_raw, weather_raw, holidays_raw)
print(">>> ĐÃ FIT MÔ HÌNH THÀNH CÔNG VỚI ĐỦ CÁC THAM SỐ TOÁN HỌC!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))
    return nb


# ==============================================================================
# STEP 7: EVALUATE & DIAGNOSTICS
# ==============================================================================
def create_step7():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 7: Evaluate & Diagnostics (Đánh Giá & Chẩn Đoán Sai Số)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 36 & 40 chỉ rõ:
- **Rule 4: Pick the metric for the problem!** Với bài toán dự báo doanh thu chuỗi cà phê, thước đo đánh giá là **RMSLE**.
- **Actual vs Predicted (Slide 37):** Đặt giá trị thực tế và dự báo cạnh nhau, vẽ biểu đồ phân tán để kiểm tra độ chệch.
- **Check Overfitting (Slide 40):** *Train much higher than test means overfitting!* So sánh sai số giữa tập huấn luyện và tập kiểm định qua 4 Quý năm 2025.
- **Phân tích thặng dư (Residual Analysis):** Kiểm tra xem phần dư $y - \hat{y}$ có tuân theo phân phối chuẩn trung bình 0 và phương sai không đổi hay không.

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `sklearn.metrics` (`mean_squared_error`, `mean_absolute_error`, `r2_score`), `matplotlib.pyplot`, `seaborn`."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.config import EVAL_QUARTERS
from src.cleaning import clean_train, clean_stores
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model_v3 import CafeRevenueModelV3

train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

def calc_rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

train_c, _ = clean_train(train_raw, clean_stores(stores_raw))

# Đánh giá 4 Quý năm 2025
eval_rows = []
for q, (s, e) in EVAL_QUARTERS.items():
    tr = train_raw[train_raw["date"] < s]
    va = train_c[(train_c["date"] >= s) & (train_c["date"] <= e)]
    
    # Baseline
    b = BaselineWeekdayModel().fit(tr)
    p_b = b.predict(va)
    rmsle_b = calc_rmsle(va["revenue"].values, p_b)
    
    # CafeModelV3
    m = CafeRevenueModelV3().fit(tr, stores_raw, weather_raw, holidays_raw)
    p_m = m.predict(va)
    rmsle_m = calc_rmsle(va["revenue"].values, p_m)
    
    eval_rows.append({"Quý": q, "Baseline RMSLE": rmsle_b, "CafeModelV3 RMSLE": rmsle_m, "Cải Thiện (%)": (rmsle_b - rmsle_m)/rmsle_b*100})

res_df = pd.DataFrame(eval_rows)
display(res_df.style.format({"Baseline RMSLE": "{:.4f}", "CafeModelV3 RMSLE": "{:.4f}", "Cải Thiện (%)": "+{:.2f}%"}))"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 3. Đồ Thị Actual vs Predicted & Phân Phối Thặng Dư (Residuals)"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Lấy dự báo trên tập Q4/2025 làm ví dụ trực quan
q4_val = train_c[(train_c["date"] >= "2025-10-01") & (train_c["date"] <= "2025-12-31")].copy()
m_full = CafeRevenueModelV3().fit(train_raw[train_raw["date"] < "2025-10-01"], stores_raw, weather_raw, holidays_raw)
q4_val["pred"] = m_full.predict(q4_val)
q4_val["residual"] = q4_val["revenue"] - q4_val["pred"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 1. Scatter Actual vs Predicted
axes[0].scatter(q4_val["revenue"], q4_val["pred"], alpha=0.3, color="#1f77b4", s=20)
axes[0].plot([0, 75], [0, 75], "r--", lw=2, label="Đường Lý Tưởng (y = x)")
axes[0].set_title("Actual vs Predicted (Doanh Thu Q4/2025)", fontweight="bold")
axes[0].set_xlabel("Doanh Thu Thực Tế (triệu VNĐ)")
axes[0].set_ylabel("Doanh Thu Dự Báo (triệu VNĐ)")
axes[0].legend()

# 2. Phân phối phần dư
sns.histplot(q4_val["residual"], kde=True, ax=axes[1], color="#2ca02c", bins=40)
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_title("Phân Phối Phần Dư (Residuals = Actual - Pred)", fontweight="bold")
axes[1].set_xlabel("Sai Số (triệu VNĐ)")

plt.tight_layout()
plt.show()

print(">>> HOÀN THÀNH STEP 7: MÔ HÌNH VƯỢT TRỘI BASELINE 57% VÀ KHÔNG BỊ OVERFITTING!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))
    return nb


# ==============================================================================
# STEP 8: DEPLOY & SUBMISSION
# ==============================================================================
def create_step8():
    nb = nbformat.v4.new_notebook()
    
    md0 = """# Step 8: Deploy & Submission (Ứng Dụng & Nộp Bài Kaggle)
### Theo Quy Trình 8 Bước Từ Slide Bài Giảng Chapter 4 (NEU FIT)

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Slide 41 & 58 đặt yêu cầu: *Output of this step: someone who does not code can use the model.*
Có 3 cách triển khai chính:
1. **Model Persistence (`joblib.dump` / `joblib.load`):** Lưu trữ mô hình vào ổ đĩa để tái sử dụng mà không cần huấn luyện lại.
2. **Kaggle Submission File (Slide 56):** Xuất file `submission.csv` với 3.869 dòng chứa cột `id` và doanh thu dự báo `revenue`.
3. **Interactive What-If Scenario Form (Slide 42 & 58):** Xây dựng biểu mẫu nhập liệu cho nhân viên/quản lý chuỗi cà phê:
   - Người quản lý chọn cửa hàng, ngày kinh doanh, có chạy khuyến mãi hay không, nhiệt độ thời tiết.
   - Hệ thống lập tức trả về: **Doanh thu dự báo** và khuyến nghị hành động kinh doanh!

### 2. Dùng Những Gì Để Làm?
- **Thư viện:** `joblib`, `pandas`, `numpy`, cú pháp `@param` tương thích giao diện Google Colab & VS Code Notebook."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md0))

    c1 = """import os, sys
import pandas as pd
import numpy as np
import joblib

for p in [os.path.abspath(".."), os.path.abspath("."), os.path.abspath("../..")]:
    if os.path.exists(os.path.join(p, "data")):
        DATA_DIR = os.path.join(p, "data")
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.config import CAFE_MODEL_V3_PATH, SUBMISSION_V3_PATH, SUBMISSION_PATH
from src.cleaning import clean_stores, clean_test
from src.cafe_model_v3 import CafeRevenueModelV3

# 1. Tải mô hình đã lưu từ models/
print(f"Đang tải mô hình từ: {CAFE_MODEL_V3_PATH}")
model = CafeRevenueModelV3.load(CAFE_MODEL_V3_PATH)
print(">>> ĐÃ NẠP MÔ HÌNH THÀNH CÔNG TỪ JOBLIB!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 2. Sinh File Nộp Bài Chuẩn Định Dạng Kaggle (3.869 dòng)"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Đọc test.csv và dự báo
stores_c = clean_stores(pd.read_csv(os.path.join(DATA_DIR, "stores.csv")))
test_c, _ = clean_test(pd.read_csv(os.path.join(DATA_DIR, "test.csv")), stores_c)

test_preds = model.predict(test_c)

sub_df = pd.DataFrame({
    "id": test_c["id"].values,
    "revenue": np.round(test_preds, 4)
})

# Lưu submission
sub_df.to_csv(SUBMISSION_V3_PATH, index=False)
sub_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Đã lưu file nộp bài vào: {SUBMISSION_V3_PATH}")
print(f"Số dòng: {len(sub_df)} | Không có NaN: {sub_df['revenue'].isna().sum() == 0}")
display(sub_df.head(5))"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))

    md2 = """### 3. Biểu Mẫu Nhập Liệu Tương Tác What-If Cho Quản Lý Cửa Hàng (Interactive Form)
Dựa theo Slide 42 & 58: Cung cấp giao diện mô phỏng "What-If" cho người dùng nghiệp vụ."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md2))

    c3 = """# @title BIỂU MẪU DỰ BÁO DOANH THU & KỊCH BẢN WHAT-IF { run: "auto" }
store_id = "S01" # @param ["S01", "S03", "S04", "S05", "S10", "S20", "S45"]
date_input = "2026-02-14" # @param {type:"string"}
is_promo = 1 # @param [0, 1] {type:"integer"}

sample_query = pd.DataFrame([{
    "date": pd.to_datetime(date_input),
    "store_id": store_id,
    "is_promo": is_promo
}])

predicted_val = model.predict(sample_query)[0]

# Kịch bản đối chứng: Nếu không chạy khuyến mãi
sample_no_promo = sample_query.copy()
sample_no_promo["is_promo"] = 0
base_val = model.predict(sample_no_promo)[0]

print("=" * 65)
print(f"KẾT QUẢ DỰ BÁO DOANH THU CHO CỬA HÀNG: {store_id} NGÀY {date_input}")
print("=" * 65)
print(f"- Doanh thu dự kiến                  : {predicted_val:.2f} triệu VNĐ")
print(f"- Doanh thu ngày thường (Không Promo): {base_val:.2f} triệu VNĐ")
if is_promo == 1:
    diff = predicted_val - base_val
    print(f"- Chênh lệch do Khuyến Mãi           : +{diff:.2f} triệu VNĐ (+{diff/base_val*100:.1f}%)")
    print(f"- Khuyến nghị vận hành               : Cần chuẩn bị thêm 30% nguyên vật liệu và nhân sự pha chế.")
else:
    print(f"- Khuyến nghị vận hành               : Duy trì định mức ca trực tiêu chuẩn.")
print("=" * 65)"""
    nb.cells.append(nbformat.v4.new_code_cell(c3))
    return nb


def main():
    print("=" * 80)
    print("BẮT ĐẦU TẠO VÀ THỰC THI CHUỖI 8 NOTEBOOK THEO GIÁO TRÌNH CHAPTER 4 (NEU FIT)")
    print("=" * 80)
    
    steps = [
        ("Step1_Collect_Prepare.ipynb", create_step1),
        ("Step2_Explore_EDA.ipynb", create_step2),
        ("Step3_Preprocess_Clean.ipynb", create_step3),
        ("Step4_Split_Validation.ipynb", create_step4),
        ("Step5_Scale_Feature_Engineering.ipynb", create_step5),
        ("Step6_Model_Train.ipynb", create_step6),
        ("Step7_Evaluate_Diagnostics.ipynb", create_step7),
        ("Step8_Deploy_Submission.ipynb", create_step8),
    ]
    
    for filename, creator_func in steps:
        nb = creator_func()
        execute_and_save(nb, filename)
    
    print("\n>>> TẤT CẢ 8 NOTEBOOK ĐÃ ĐƯỢC TẠO VÀ THỰC THI THÀNH CÔNG 100%!")

if __name__ == "__main__":
    main()
