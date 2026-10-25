"""
src/cleaning.py
Xử lý làm sạch và chuẩn hóa dữ liệu cho bài toán Cafe Revenue Prediction:
1. clean_stores: sửa lỗi chính tả quận (District), điền khuyết diện tích (area_m2) bằng tỷ lệ median theo loại quán.
2. clean_weather: chuyển đổi nhiệt độ Fahrenheit sang Celsius, nội suy chuỗi thời tiết đầy đủ.
3. clean_holidays: tách ngày lễ quốc gia (non-Tet) và sự kiện, lập bản đồ ngày Tết Âm lịch.
4. clean_train: loại bỏ trùng lặp, khắc phục lỗi nhân 1000x, 10x và chia 10x do lỗi nhập liệu.
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd


def clean_stores(stores_raw: pd.DataFrame) -> pd.DataFrame:
    """Làm sạch và chuẩn hóa dữ liệu cửa hàng."""
    s = stores_raw.copy()
    s["store_type"] = s["store_type"].str.lower().str.strip()
    s["district"] = s["district"].str.strip().replace({
        "Q.10": "District 10",
        "Quan 3": "District 3",
        "Quan 7": "District 7",
    })
    s["open_date"] = pd.to_datetime(s["open_date"], format="mixed", dayfirst=True)
    
    # Điền khuyết diện tích bằng tỷ lệ median m2/ghế theo loại quán
    ratio_by_type = (s["area_m2"] / s["seats"]).groupby(s["store_type"]).median()
    for idx, row in s[s["area_m2"].isna()].iterrows():
        s.loc[idx, "area_m2"] = round(row["seats"] * ratio_by_type[row["store_type"]], 1)
    return s


def clean_weather(weather_raw: pd.DataFrame, cities: list) -> pd.DataFrame:
    """Chuẩn hóa nhiệt độ F -> C và nội suy lưới thời tiết đầy đủ cho tất cả các thành phố."""
    w = weather_raw.copy()
    w["date"] = pd.to_datetime(w["date"])
    
    # Sửa lỗi nhiệt độ nhập theo thang độ F (> 45°C)
    f_mask = w["temp_c"] > 45.0
    w.loc[f_mask, "temp_c"] = (w.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0
    
    # Đánh dấu các ngày có số đo thời tiết thực tế (không bị khuyết)
    exact_w = w.dropna(subset=["temp_c", "rain_mm"])[["date", "city"]].drop_duplicates()
    exact_w["has_exact_weather"] = True

    # Tạo lưới ngày đầy đủ 2024-01-01 đến 2026-03-31
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", "2026-03-31"), sorted(cities)],
        names=["date", "city"],
    ).to_frame().reset_index(drop=True)
    
    w = grid.merge(w, on=["date", "city"], how="left").sort_values(["city", "date"]).reset_index(drop=True)
    w["temp_c"] = w.groupby("city")["temp_c"].transform(lambda x: x.interpolate().bfill().ffill())
    w["rain_mm"] = w["rain_mm"].fillna(0.0)
    w = w.merge(exact_w, on=["date", "city"], how="left")
    w["has_exact_weather"] = w["has_exact_weather"].fillna(False)
    return w


def clean_holidays(holidays_raw: pd.DataFrame) -> Tuple[pd.Series, pd.Series, Dict[int, pd.Timestamp]]:
    """Tách danh sách ngày lễ quốc gia, ngày sự kiện và từ điển mốc Tết Nguyên Đán."""
    h = holidays_raw.copy()
    h["date"] = pd.to_datetime(h["date"])
    h_dedup = h.drop_duplicates(subset=["date"], keep="first").copy()
    
    non_tet_nat = h_dedup[(h_dedup["type"] == "national_holiday") & (~h_dedup["holiday_name"].str.contains("Tet"))]
    nat_hol = non_tet_nat.groupby("date")["holiday_name"].first().rename("nat_holiday")
    
    evt_hol = (
        h_dedup[(h_dedup["type"] == "event") & (~h_dedup["holiday_name"].str.contains("Tet"))]
        .groupby("date")["holiday_name"]
        .first()
        .rename("evt_holiday")
    )
    
    tet_rows = h[h["holiday_name"] == "Lunar New Year (Tet)"]
    tet_dates = dict(zip(tet_rows["date"].dt.year, tet_rows["date"]))
    return nat_hol, evt_hol, tet_dates


def clean_train(df_in: pd.DataFrame, stores_clean: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Làm sạch các bất thường và lỗi ghi nhận trong train.csv."""
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date", "store_id"]).copy()
    df = df.merge(stores_clean[["store_id", "open_date"]], on="store_id", how="left")
    df["dow"] = df["date"].dt.dayofweek
    df["days_since_open"] = (df["date"] - df["open_date"]).dt.days

    # 1. Sửa lỗi gõ thêm 3 số không (nhân 1000x: revenue > 1000 triệu)
    n_1000x = int((df["revenue"] > 1000).sum())
    df.loc[df["revenue"] > 1000, "revenue"] /= 1000.0

    # 2. Sửa lỗi sai số thập phân: nhân 10x hoặc chia 10x
    med_dow = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    ratio_dow = df["revenue"] / med_dow
    mask_10x = (df["days_since_open"] > 14) & (ratio_dow > 3.0)
    mask_01x = ratio_dow < 0.20
    df.loc[mask_10x, "revenue"] /= 10.0
    df.loc[mask_01x, "revenue"] *= 10.0
    
    stats = {
        "duplicates_removed": len(df_in) - len(df),
        "fixed_1000x": n_1000x,
        "fixed_10x": int(mask_10x.sum()),
        "fixed_01x": int(mask_01x.sum()),
    }
    return df[["date", "store_id", "is_promo", "revenue"]].reset_index(drop=True), stats
