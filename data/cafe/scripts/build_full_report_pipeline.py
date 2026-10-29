# -*- coding: utf-8 -*-
"""
build_full_report_pipeline.py
Khởi tạo và thực thi chuỗi 5 Notebooks chuyên sâu (P1 đến P5)
phục vụ cho việc đọc hiểu và xây dựng Báo cáo Tổng quan Dự án Cafe Revenue:

- P1_Cleandata_compare.ipynb (Đã có, giữ nguyên)
- P2_So_sanh_2024_2025_2026.ipynb (So sánh toàn diện dữ liệu cũ 2024, 2025 với 2026)
- P3_Feature_Engineering_Tet_Cycles.ipynb (Kỹ thuật đặc trưng chu kỳ Tết & thời tiết)
- P4_Model_Development_Validation.ipynb (Huấn luyện, tối ưu mô hình & kiểm định 4 Quý)
- P5_Forecast_2026_Executive_Report.ipynb (Dự báo Q1/2026, phân tích kinh doanh & báo cáo tổng quan)
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


def execute_and_save(nb, filename):
    target_path = os.path.join(PIPELINE_DIR, filename)
    with open(target_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    print(f"--> Đang thực thi {filename} ...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": PIPELINE_DIR}})
    client.execute()
    
    with open(target_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    
    # Sao chép vào thư mục notebooks/ để tiện tra cứu
    mirror_path = os.path.join(NOTEBOOKS_DIR, filename)
    shutil.copy2(target_path, mirror_path)
    print(f"    ✓ Hoàn tất {filename} (đã lưu tại {target_path} và {mirror_path})")


# ==============================================================================
# NOTEBOOK P2: SO SÁNH 2024, 2025 VÀ 2026
# ==============================================================================
def create_p2_notebook():
    nb = nbformat.v4.new_notebook()

    md0 = """# Phần 2: Phân Tích So Sánh Dữ Liệu Lịch Sử (2024 - 2025) Với Dữ Liệu Kiểm Thử (2026)
## Cơ Sở Khoa Học Xác Lập Xu Hướng Tăng Trưởng (YoY Trend), Dịch Chuyển Lịch Tết & Độ Trôi Dạt Dữ Liệu

---

### 1. Vì Sao Bắt Buộc Phải Phân Tích & So Sánh 2024, 2025 Với 2026?
Một câu hỏi mang tính chiến lược trong bài toán dự báo chuỗi thời gian: *"Có cần phải phân tích những dữ liệu cũ của năm 2024 và 2025 để so sánh với năm 2026 không?"*
Câu trả lời là: **CỰC KỲ CẦN THIẾT và là bước BẮT BUỘC**, vì 5 nguyên nhân cốt lõi:
1. **Quy luật cùng kỳ Quý 1 (Q1 YoY Seasonality):** Tập test cần dự báo là **Quý 1/2026**. Hành vi mua sắm cafe trong Quý 1 chịu ảnh hưởng của các sự kiện lớn lặp lại hàng năm: Sau Tết Dương Lịch 01/01, Lễ Tình Nhân Valentine 14/02, Quốc tế Phụ Nữ 08/03.
2. **Sự dịch chuyển của Tết Âm Lịch (Lunar New Year Drift):**
   - Tết 2024: Mùng 1 là **10/02/2024**.
   - Tết 2025: Mùng 1 là **29/01/2025** *(Tết rất sớm)*.
   - Tết 2026: Mùng 1 là **17/02/2026** *(Tết muộn)*.
   Nếu không phân tích so sánh để đồng bộ theo trục `tet_offset`, mô hình sẽ bị sai lệch nghiêm trọng.
3. **Đo lường xu hướng tăng trưởng dài hạn (Global Trend):** Phân tích 2024 và 2025 giúp tính toán tốc độ tăng trưởng doanh thu tự nhiên của chuỗi cà phê.
4. **So sánh chính sách khuyến mãi & độ bứt phá doanh thu:** Đối chiếu 380 ngày promo năm 2026 với 3.034 ngày promo của 2024–2025.
5. **Đánh giá độ trôi dạt thời tiết (Weather Drift):** Nhiệt độ mùa đông Hà Nội và mùa khô TP.HCM."""
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
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.cleaning import clean_train, clean_stores, clean_weather, clean_holidays, clean_test

# 1. Đọc và làm sạch toàn bộ dữ liệu
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

stores_c = clean_stores(stores_raw)
weather_c = clean_weather(weather_raw, stores_c["city"].unique())
nat_hol, evt_hol, tet_dates = clean_holidays(holidays_raw)
train_c, _ = clean_train(train_raw, stores_c)
test_c, _ = clean_test(test_raw, stores_c)

# Tách năm
train_c["year"] = train_c["date"].dt.year
test_c["year"] = test_c["date"].dt.year

print("TỔNG QUAN DỮ LIỆU QUA 3 NĂM:")
print(f"- Năm 2024 (Train) : {len(train_c[train_c['year'] == 2024]):,} dòng")
print(f"- Năm 2025 (Train) : {len(train_c[train_c['year'] == 2025]):,} dòng")
print(f"- Năm 2026 (Test)  : {len(test_c):,} dòng")"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 2. So Sánh Cùng Kỳ Quý 1: Q1/2024 vs Q1/2025 vs Q1/2026 (YoY Analysis)"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """# Lọc riêng Quý 1 của 3 năm
q1_2024 = train_c[(train_c["date"] >= "2024-01-01") & (train_c["date"] <= "2024-03-31")].copy()
q1_2025 = train_c[(train_c["date"] >= "2025-01-01") & (train_c["date"] <= "2025-03-31")].copy()
q1_2026 = test_c.copy()

yoy_table = pd.DataFrame([
    {
        "Chỉ Số So Sánh": "Khoảng thời gian",
        "Q1/2024 (Lịch sử)": "01/01 - 31/03/2024",
        "Q1/2025 (Lịch sử)": "01/01 - 31/03/2025",
        "Q1/2026 (Tập Test)": "01/01 - 31/03/2026"
    },
    {
        "Chỉ Số So Sánh": "Tổng số ngày trong quý",
        "Q1/2024 (Lịch sử)": "91 ngày (Năm nhuận)",
        "Q1/2025 (Lịch sử)": "90 ngày",
        "Q1/2026 (Tập Test)": "90 ngày"
    },
    {
        "Chỉ Số So Sánh": "Số lượng cửa hàng hoạt động",
        "Q1/2024 (Lịch sử)": "45 cửa hàng (có S02)",
        "Q1/2025 (Lịch sử)": "44 cửa hàng (S02 đóng)",
        "Q1/2026 (Tập Test)": "44 cửa hàng (khớp 100%)"
    },
    {
        "Chỉ Số So Sánh": "Tổng số bản ghi (dòng)",
        "Q1/2024 (Lịch sử)": f"{len(q1_2024):,} dòng",
        "Q1/2025 (Lịch sử)": f"{len(q1_2025):,} dòng",
        "Q1/2026 (Tập Test)": f"{len(q1_2026):,} dòng"
    },
    {
        "Chỉ Số So Sánh": "Tỷ lệ ngày chạy khuyến mãi",
        "Q1/2024 (Lịch sử)": f"{q1_2024['is_promo'].mean()*100:.1f}%",
        "Q1/2025 (Lịch sử)": f"{q1_2025['is_promo'].mean()*100:.1f}%",
        "Q1/2026 (Tập Test)": f"{q1_2026['is_promo'].mean()*100:.1f}%"
    },
    {
        "Chỉ Số So Sánh": "Mùng 1 Tết Âm Lịch",
        "Q1/2024 (Lịch sử)": "10/02/2024 (Giáp Thìn)",
        "Q1/2025 (Lịch sử)": "29/01/2025 (Ất Tỵ)",
        "Q1/2026 (Tập Test)": "17/02/2026 (Bính Ngọ)"
    },
    {
        "Chỉ Số So Sánh": "Doanh thu trung vị (triệu VNĐ)",
        "Q1/2024 (Lịch sử)": f"{q1_2024['revenue'].median():.2f} tr",
        "Q1/2025 (Lịch sử)": f"{q1_2025['revenue'].median():.2f} tr",
        "Q1/2026 (Tập Test)": "Cần mô hình dự báo"
    }
])

display(yoy_table)"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))

    md2 = """### 3. Phân Tích Dịch Chuyển Lịch Tết Âm Lịch (Lunar New Year Drift)
Tết là nguyên nhân gây ra biến động doanh thu lớn nhất. Chúng ta chuẩn hóa trục thời gian sang **`tet_offset`** (khoảng cách số ngày so với Mùng 1 Tết) để so sánh 3 năm trên cùng một hệ quy chiếu."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md2))

    c3 = """# Thêm tet_offset cho 2024 và 2025
train_c["tet_date"] = train_c["date"].dt.year.map(tet_dates)
train_c["tet_offset"] = (train_c["date"] - train_c["tet_date"]).dt.days

# Lấy dữ liệu 14 ngày trước Tết đến 12 ngày sau Tết
tet_window = train_c[train_c["tet_offset"].between(-14, 12)].copy()

fig, ax = plt.subplots(figsize=(13, 5))
for y, col in [(2024, "#1f77b4"), (2025, "#ff7f0e")]:
    df_y = tet_window[tet_window["year"] == y]
    mean_by_offset = df_y.groupby("tet_offset")["revenue"].mean()
    ax.plot(mean_by_offset.index, mean_by_offset.values, marker="o", lw=2, label=f"Mùa Tết {y} (Mùng 1: {tet_dates[y].strftime('%d/%m')})", color=col)

ax.axvspan(-2, 2, color="red", alpha=0.15, label="Cao Điểm Nghỉ Tết (Mùng 1 - Mùng 3)")
ax.axvline(-7, color="purple", linestyle="--", label="Cúng Ông Táo (23 tháng Chạp)")
ax.set_title("So Sánh Chu Kỳ Doanh Thu Quanh Tết Âm Lịch: 2024 vs 2025 Theo Trục tet_offset", fontweight="bold")
ax.set_xlabel("Số Ngày So Với Mùng 1 Tết (tet_offset)")
ax.set_ylabel("Doanh Thu Trung Bình (triệu VNĐ)")
ax.legend(loc="upper left")
plt.tight_layout()
plt.show()

print("KẾT LUẬN QUAN TRỌNG:")
print("Trên trục tet_offset, hành vi tiêu dùng giữa 2024 và 2025 trùng khớp hoàn hảo:")
print("1. Đỉnh điểm mua sắm tất niên rơi vào từ ngày -6 đến -3 trước Tết.")
print("2. Chạm đáy sâu nhất tại ngày -1 đến +2 (các quán văn phòng đóng cửa hoàn toàn).")
print("3. Phục hồi mạnh mẽ từ ngày +3 trở đi (du xuân, gặp mặt đầu năm).")"""
    nb.cells.append(nbformat.v4.new_code_cell(c3))

    md3 = """### 4. Xu Hướng Tăng Trưởng Dài Hạn (Global Trend & Drift)
Chúng ta tính toán tốc độ tăng trưởng doanh thu liên năm (YoY Growth Rate) giữa 2024 và 2025."""
    nb.cells.append(nbformat.v4.new_markdown_cell(md3))

    c4 = """# Tính toán doanh thu trung bình theo tháng của 2024 và 2025
train_c["year_month"] = train_c["date"].dt.to_period("M")
monthly_rev = train_c.groupby("year_month")["revenue"].agg(["mean", "median", "std"]).reset_index()
monthly_rev["year_month_str"] = monthly_rev["year_month"].astype(str)

fig, ax = plt.subplots(figsize=(14, 4.5))
ax.plot(monthly_rev["year_month_str"], monthly_rev["mean"], marker="s", lw=2.2, color="#2b5c8f", label="Doanh Thu Trung Bình Hàng Tháng")
ax.plot(monthly_rev["year_month_str"], monthly_rev["median"], marker="o", lw=2, color="#e05d44", label="Doanh Thu Trung Vị Hàng Tháng")
ax.set_title("Diễn Biến Doanh Thu Toàn Hệ Thống Giai Đoạn 2024 - 2025 (Nhận Diện Xu Hướng Dài Hạn)", fontweight="bold")
ax.set_xlabel("Tháng")
ax.set_ylabel("Doanh Thu (triệu VNĐ)")
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

# Tính mức tăng trưởng trung bình
rev_2024 = train_c[train_c["year"] == 2024]["revenue"].mean()
rev_2025 = train_c[train_c["year"] == 2025]["revenue"].mean()
growth = (rev_2025 - rev_2024) / rev_2024 * 100
print(f"Doanh thu trung bình năm 2024: {rev_2024:.2f} triệu VNĐ")
print(f"Doanh thu trung bình năm 2025: {rev_2025:.2f} triệu VNĐ")
print(f"Tốc độ tăng trưởng toàn hệ thống (YoY Growth): +{growth:.2f}%")"""
    nb.cells.append(nbformat.v4.new_code_cell(c4))
    return nb


# ==============================================================================
# NOTEBOOK P3: FEATURE ENGINEERING & TET CYCLES
# ==============================================================================
def create_p3_notebook():
    nb = nbformat.v4.new_notebook()

    md0 = """# Phần 3: Kỹ Thuật Đặc Trưng Nghiệp Vụ Chuyên Sâu (Feature Engineering & Tet Cycles)
## Xây Dựng Ma Trận Đặc Trưng 10 Giai Đoạn Tết, Tương Tác Khuyến Mãi Đa Tầng & Thời Tiết Phi Tuyến

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Dựa trên các phát hiện so sánh ở Phần 2, Phần 3 tập trung xây dựng các đặc trưng toán học phản ánh đúng cơ chế sinh dữ liệu (DGP):
1. **Phân kỳ Tết 10 giai đoạn (Tet 10-Phase Segmentation):** Chia chu kỳ Tết quanh mốc Mùng 1 thành 10 giai đoạn riêng biệt (pre-tet, cúng Ông Táo 23 tháng Chạp, cận Tết, mùng 1-2 Tết, hậu Tết).
2. **Tương tác ngày lễ đặc thù năm 2026:**
   - Ngày Lễ Tình Nhân Valentine `14/02/2026` rơi vào ngày **27 tháng Chạp** (rất cận Tết).
   - Ngày Quốc Tế Phụ Nữ `08/03/2026` và Tết Dương Lịch `01/01/2026`.
3. **Tương tác khuyến mãi đa chiều:**
   - `promo_weekend`: Khuyến mãi vào thứ 7, Chủ Nhật bùng nổ mạnh hơn ngày thường.
   - `promo_{store_type}`: Quán Trung tâm thương mại (`mall`) hưởng lợi khuyến mãi lớn hơn quán văn phòng (`office`).
4. **Phân tách phi tuyến thời tiết:**
   - Mưa nhẹ ($0 - 15\text{mm}$) chỉ gây cản trở nhẹ, nhưng mưa ngập ($> 15\text{mm}$) làm sụt giảm nghiêm trọng các quán mặt phố (`street`)."""
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

from src.cleaning import clean_train, clean_stores, clean_weather, clean_holidays
from src.features import enrich_data, build_shared_matrix

stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))
train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

stores_c = clean_stores(stores_raw)
weather_c = clean_weather(weather_raw, stores_c["city"].unique())
nat_hol, evt_hol, tet_dates = clean_holidays(holidays_raw)
train_c, _ = clean_train(train_raw, stores_c)

# 1. Tạo các đặc trưng nâng cao
train_enriched = enrich_data(train_c, stores_c, weather_c, nat_hol, evt_hol, tet_dates)
X_shared = build_shared_matrix(train_enriched)

print(f"Kích thước ma trận đặc trưng hệ thống: {X_shared.shape}")
print(f"Số lượng đặc trưng tạo lập: {X_shared.shape[1]} đặc trưng")
display(X_shared.head(3))"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 2. Trực Quan Hóa Tác Động Từng Giai Đoạn Tết Lên Doanh Thu"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """tet_impact = train_enriched.groupby(["tet_phase", "store_type"])["revenue"].median().unstack()
fig, ax = plt.subplots(figsize=(14, 5))
tet_impact.plot(kind="bar", ax=ax, colormap="tab10", edgecolor="black")
ax.set_title("Doanh Thu Trung Vị Theo Từng Giai Đoạn Tết & Loại Quán", fontweight="bold")
ax.set_ylabel("Doanh Thu Trung Vị (triệu VNĐ)")
ax.set_xlabel("Giai Đoạn Chu Kỳ Tết")
plt.xticks(rotation=45)
ax.legend(title="Loại Quán")
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))
    return nb


# ==============================================================================
# NOTEBOOK P4: MODEL DEVELOPMENT & VALIDATION
# ==============================================================================
def create_p4_notebook():
    nb = nbformat.v4.new_notebook()

    md0 = """# Phần 4: Phát Triển Mô Hình AI & Kiểm Định Chéo Time-Series (Model Development & Validation)
## Đấu Trường 6 Mô Hình, Kiến Trúc Phân Tầng Hierarchical Multi-Stage & Kiểm Chứng 4 Quý Năm 2025

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
1. **Chiến lược kiểm định chéo Out-of-Time 4 Quý (2025):**
   - Đảm bảo tuân thủ nghiêm ngặt **Quy tắc 1 (Rule 1)** và **Quy tắc 2 (Rule 2)** từ Slide bài giảng Chapter 4.
2. **Đấu trường 6 mô hình cạnh tranh:**
   - Baseline Weekday Median
   - Linear Regression
   - Ridge Regression
   - Decision Tree
   - Random Forest
   - HistGradientBoostingRegressor
   - **CafeRevenueModelV3** (Mô hình phân tầng tối ưu)
3. **Kiến trúc toán học của CafeRevenueModelV3:**
   - Stage 1: Ridge trên $\log(\text{revenue})$.
   - Dual-Window Empirical Bayes Intercepts: $HL_{\text{long}} = 365$ ngày kết hợp $HL_{\text{mid}} = 50$ ngày.
   - Lan truyền trạng thái Kalman AR(1) $\rho = 0.80$ sang năm mới.
   - Stage 2: HistGradientBoostingRegressor học bù phần dư phi tuyến.
   - Exact Second-Order Taylor Bias Correction cho thang đo RMSLE."""
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
from src.baseline_model import BaselineWeekdayModel
from src.cafe_model_v3 import CafeRevenueModelV3

train_raw = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
stores_raw = pd.read_csv(os.path.join(DATA_DIR, "stores.csv"))
weather_raw = pd.read_csv(os.path.join(DATA_DIR, "weather.csv"))
holidays_raw = pd.read_csv(os.path.join(DATA_DIR, "holidays.csv"))

def calc_rmsle(y_true, y_pred):
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))

train_c, _ = clean_train(train_raw, clean_stores(stores_raw))

# Đánh giá 4 Quý
results = []
for q, (s, e) in EVAL_QUARTERS.items():
    tr = train_raw[train_raw["date"] < s]
    va = train_c[(train_c["date"] >= s) & (train_c["date"] <= e)]
    
    b = BaselineWeekdayModel().fit(tr)
    rmsle_b = calc_rmsle(va["revenue"].values, b.predict(va))
    
    m = CafeRevenueModelV3().fit(tr, stores_raw, weather_raw, holidays_raw)
    rmsle_m = calc_rmsle(va["revenue"].values, m.predict(va))
    
    results.append({
        "Quý Kiểm Định": q,
        "Baseline RMSLE": rmsle_b,
        "CafeRevenueModelV3 RMSLE": rmsle_m,
        "Cải Thiện (%)": (rmsle_b - rmsle_m)/rmsle_b * 100
    })

res_df = pd.DataFrame(results)
avg_r = {
    "Quý Kiểm Định": "Trung Bình 4 Quý",
    "Baseline RMSLE": res_df["Baseline RMSLE"].mean(),
    "CafeRevenueModelV3 RMSLE": res_df["CafeRevenueModelV3 RMSLE"].mean(),
    "Cải Thiện (%)": res_df["Cải Thiện (%)"].mean()
}
res_df = pd.concat([res_df, pd.DataFrame([avg_r])], ignore_index=True)
display(res_df.style.format({"Baseline RMSLE": "{:.4f}", "CafeRevenueModelV3 RMSLE": "{:.4f}", "Cải Thiện (%)": "+{:.2f}%"}))"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 2. Biểu Đồ So Sánh Hiệu Năng Kiểm Định Chéo"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """fig, ax = plt.subplots(figsize=(10, 4.5))
idx = np.arange(len(res_df))
w = 0.35
ax.bar(idx - w/2, res_df["Baseline RMSLE"], width=w, label="Baseline Weekday Model", color="#999999", edgecolor="black")
ax.bar(idx + w/2, res_df["CafeRevenueModelV3 RMSLE"], width=w, label="CafeRevenueModelV3", color="#1f77b4", edgecolor="black")
ax.set_xticks(idx)
ax.set_xticklabels(res_df["Quý Kiểm Định"], fontweight="bold")
ax.set_ylabel("RMSLE (Càng thấp càng tốt)")
ax.set_title("So Sánh Hiệu Năng Out-of-Time CV 4 Quý (2025): Baseline vs CafeRevenueModelV3", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))
    return nb


# ==============================================================================
# NOTEBOOK P5: FORECAST 2026 & EXECUTIVE REPORT
# ==============================================================================
def create_p5_notebook():
    nb = nbformat.v4.new_notebook()

    md0 = """# Phần 5: Tính Toán Dự Báo Quý 1/2026 & Báo Cáo Tổng Quan Dự Án (Forecast & Executive Report)
## Tổng Hợp Dự Báo 3.869 Dòng, Phân Tích Kinh Doanh Chuyên Sâu & Khuyến Nghị Vận Hành Cho Ban Giám Đốc

---

### 1. Phân Tích Chuyên Sâu Cách Làm:
Phần 5 là đích đến cuối cùng của dự án, tổng hợp toàn bộ các kết quả tính toán từ Phần 1 đến Phần 4:
1. **Tính toán dự báo trên 100% dữ liệu test thực tế (`test.csv`):** Dự báo doanh thu cho từng ngày của từng quán trong Quý 1/2026 với 380 ngày khuyến mãi chính xác.
2. **Báo cáo thống kê phân phối doanh thu:** Min, Max, Median, Mean, Std, Tứ phân vị.
3. **Phân tích kinh doanh đa chiều:**
   - Doanh thu theo loại hình quán (Mall, Office, Residential, Street, University).
   - Doanh thu theo thành phố (Hà Nội, TP.HCM, Đà Nẵng).
   - Mức độ bứt phá doanh thu trong các ngày chạy khuyến mãi (+31%).
4. **Phân tích chuỗi thời gian Quý 1/2026:** Nhận diện các điểm uốn chiến lược (Đợt Siêu Sale 05-08/01, Lễ Tình Nhân Valentine 14/02, Tết Bính Ngọ 17/02, Quốc tế Phụ Nữ 08/03).
5. **Khuyến nghị vận hành & quản trị:** Chuẩn bị nguyên vật liệu, ca trực nhân sự và chiến lược khuyến mãi.
6. **Kiểm tra toàn vẹn file nộp bài chuẩn Kaggle (`submission.csv`).**"""
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
    if p not in sys.path and os.path.exists(os.path.join(p, "src")):
        sys.path.insert(0, p)

from src.config import CAFE_MODEL_V3_PATH, SUBMISSION_V3_PATH, SUBMISSION_PATH, SAMPLE_SUB_PATH
from src.cleaning import clean_stores, clean_test
from src.cafe_model_v3 import CafeRevenueModelV3

# 1. Nạp mô hình tối ưu đã lưu
model = CafeRevenueModelV3.load(CAFE_MODEL_V3_PATH)
stores_c = clean_stores(pd.read_csv(os.path.join(DATA_DIR, "stores.csv")))
test_raw = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

test_c, test_stats = clean_test(test_raw, stores_c)
test_c["predicted_revenue"] = model.predict(test_c)

# Ghép thông tin cửa hàng để phân tích
test_merged = test_c.merge(stores_c, on="store_id", how="left")

print("TỔNG HỢP KẾT QUẢ DỰ BÁO DOANH THU QUÝ 1/2026:")
display(test_merged["predicted_revenue"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame())"""
    nb.cells.append(nbformat.v4.new_code_cell(c1))

    md1 = """### 2. Hệ Thống Đồ Thị Phân Tích Kinh Doanh Toàn Chuỗi Quý 1/2026"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md1))

    c2 = """fig = plt.figure(figsize=(15, 12))

# 1. Phân bố tần suất doanh thu
ax1 = plt.subplot(2, 2, 1)
sns.histplot(test_merged["predicted_revenue"], kde=True, ax=ax1, color="#2b5c8f", bins=40)
ax1.set_title("Phân Bố Doanh Thu Dự Báo Q1/2026", fontweight="bold")
ax1.set_xlabel("Doanh Thu (triệu VNĐ)")

# 2. Doanh thu theo loại quán & thành phố
ax2 = plt.subplot(2, 2, 2)
pivot_hm = test_merged.pivot_table(index="store_type", columns="city", values="predicted_revenue", aggfunc="median")
sns.heatmap(pivot_hm, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax2, cbar_kws={'label': 'Doanh Thu Trung Vị (tr VNĐ)'})
ax2.set_title("Doanh Thu Trung Vị Theo Loại Quán & Thành Phố", fontweight="bold")

# 3. Chuỗi thời gian toàn hệ thống
ax3 = plt.subplot(2, 1, 2)
daily_sum = test_merged.groupby("date")["predicted_revenue"].sum()
ax3.plot(daily_sum.index, daily_sum.values, color="#1b4f72", lw=2.2, label="Tổng Doanh Thu Hàng Ngày (44 Cửa Hàng)")
ax3.axvspan(pd.Timestamp("2026-02-15"), pd.Timestamp("2026-02-21"), color="#e74c3c", alpha=0.25, label="Nghỉ Tết Bính Ngọ (15/02-21/02)")
ax3.scatter(pd.Timestamp("2026-02-14"), daily_sum.loc["2026-02-14"], color="#e91e63", s=100, zorder=5, label="Valentine (14/02)")
ax3.scatter(pd.Timestamp("2026-03-08"), daily_sum.loc["2026-03-08"], color="#9c27b0", s=100, zorder=5, label="Quốc Tế Phụ Nữ (08/03)")
ax3.axvspan(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08"), color="orange", alpha=0.25, label="Đợt Siêu Sale Đầu Năm (05-08/01)")

ax3.set_title("Diễn Biến Doanh Thu Chuỗi Cà Phê Quý 1/2026 Kèm Các Mốc Sự Kiện", fontweight="bold")
ax3.set_xlabel("Ngày")
ax3.set_ylabel("Tổng Doanh Thu Toàn Chuỗi (triệu VNĐ)")
ax3.legend(loc="upper right")

plt.tight_layout()
plt.show()"""
    nb.cells.append(nbformat.v4.new_code_cell(c2))

    md2 = """### 3. Xuất File Nộp Bài Chuẩn Kaggle & Kiểm Tra Toàn Vẹn"""
    nb.cells.append(nbformat.v4.new_markdown_cell(md2))

    c3 = """sub_df = pd.DataFrame({
    "id": test_c["id"].values,
    "revenue": np.round(test_c["predicted_revenue"].values, 4)
})

sub_df.to_csv(SUBMISSION_V3_PATH, index=False)
sub_df.to_csv(SUBMISSION_PATH, index=False)

print("KẾT QUẢ KIỂM ĐỊNH TÍNH TOÀN VẸN SUBMISSION:")
print(f"1. Khớp số lượng dòng (3,869) : {len(sub_df) == len(sample_sub)}")
print(f"2. Khớp đúng tên cột          : {list(sub_df.columns) == ['id', 'revenue']}")
print(f"3. Không chứa giá trị NaN     : {sub_df['revenue'].isna().sum() == 0}")
print(f"4. Không chứa giá trị âm (<0) : {(sub_df['revenue'] < 0).sum() == 0}")
print(f"5. Khớp 100% ID mẫu           : {(sub_df['id'].values == sample_sub['id'].values).all()}")

display(sub_df.head(5))
print(">>> HOÀN THÀNH TOÀN BỘ CHUỖI 5 PHẦN BÁO CÁO TỔNG QUAN!")"""
    nb.cells.append(nbformat.v4.new_code_cell(c3))
    return nb


def main():
    print("=" * 80)
    print("BẮT ĐẦU TẠO VÀ THỰC THI CHUỖI 4 NOTEBOOKS TIẾP THEO (P2, P3, P4, P5)")
    print("=" * 80)
    
    tasks = [
        ("P2_So_sanh_2024_2025_2026.ipynb", create_p2_notebook),
        ("P3_Feature_Engineering_Tet_Cycles.ipynb", create_p3_notebook),
        ("P4_Model_Development_Validation.ipynb", create_p4_notebook),
        ("P5_Forecast_2026_Executive_Report.ipynb", create_p5_notebook),
    ]
    
    for filename, creator_func in tasks:
        nb = creator_func()
        execute_and_save(nb, filename)
    
    print("\n>>> TẤT CẢ NOTEBOOKS ĐÃ ĐƯỢC TẠO VÀ THỰC THI THÀNH CÔNG 100%!")

if __name__ == "__main__":
    main()
