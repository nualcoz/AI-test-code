"""
src/features.py
Feature Engineering cho mô hình Cafe Revenue:
1. enrich_data: kết hợp thông tin cửa hàng, thời tiết, ngày lễ, và tính chu kỳ Tết Âm lịch.
2. build_shared_matrix: xây dựng ma trận đặc trưng cho các hiệu ứng chung (Khuyến mãi, thời tiết tuyến tính, ngày trong tuần, lễ Tết, khai trương, xu hướng).
3. build_gbdt_matrix: tạo tập đặc trưng số và phân loại cho bộ tăng cường HistGradientBoostingRegressor.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def enrich_data(
    df_in: pd.DataFrame,
    stores_clean: pd.DataFrame,
    weather_clean: pd.DataFrame,
    nat_hol: pd.Series,
    evt_hol: pd.Series,
    tet_dates: Dict[int, pd.Timestamp],
) -> pd.DataFrame:
    """Nối thông tin cửa hàng, thời tiết, ngày lễ và phân kỳ Tết Âm lịch."""
    df = df_in.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.merge(stores_clean, on="store_id", how="left").merge(
        weather_clean, on=["date", "city"], how="left"
    )
    df = df.merge(nat_hol, on="date", how="left").merge(evt_hol, on="date", how="left")
    df["nat_holiday"] = df["nat_holiday"].fillna("none")
    df["evt_holiday"] = df["evt_holiday"].fillna("none")
    df["tet_date"] = df["date"].dt.year.map(tet_dates)
    df["tet_offset"] = (df["date"] - df["tet_date"]).dt.days

    # Phân kỳ giai đoạn Tết chính xác theo đặc tính thực tế và văn hóa Việt Nam
    df["tet_phase"] = "none"
    df.loc[df["tet_offset"].between(-14, -13), "tet_phase"] = "pre_tet_14_13"
    df.loc[df["tet_offset"].between(-12, -11), "tet_phase"] = "pre_tet_12_11"
    df.loc[df["tet_offset"].between(-10, -8), "tet_phase"] = "pre_tet_10_8"
    df.loc[df["tet_offset"] == -7, "tet_phase"] = "tet_23_ong_tao"
    df.loc[df["tet_offset"].between(-6, -3), "tet_phase"] = "pre_tet_6_3"
    df.loc[df["tet_offset"].between(-2, 1), "tet_phase"] = "tet_core_m2_p1"
    df.loc[df["tet_offset"] == 2, "tet_phase"] = "tet_p2"
    df.loc[df["tet_offset"].between(3, 7), "tet_phase"] = "post_tet_3_7"
    df.loc[df["tet_offset"].between(8, 9), "tet_phase"] = "post_tet_8_9"
    df.loc[df["tet_offset"].between(10, 12), "tet_phase"] = "post_tet_10_12"
    return df


def build_shared_matrix(df: pd.DataFrame, shared_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Tạo ma trận biến giải thích cho các hiệu ứng toàn hệ thống (Shared Multiplicative Effects)."""
    st = df["store_type"]
    city = df["city"]
    dow = df["date"].dt.dayofweek.astype(str)
    is_nat = (df["nat_holiday"] != "none").astype(str)
    is_evt = (df["evt_holiday"] != "none").astype(str)

    # Hiệu ứng phân loại tương tác giữa loại quán và biến thời gian/sự kiện
    cat_df = pd.DataFrame(
        {
            "st_dow": st + "_" + dow,
            "st_is_nat": st + "_" + is_nat,
            "st_is_evt": st + "_" + is_evt,
            "st_tet": st + "_" + df["tet_phase"],
        },
        index=df.index,
    )
    X_cat = pd.get_dummies(cat_df, dtype=float)

    # Hiệu ứng số: Khai trương, mùa hè sinh viên, xu hướng tăng trưởng, mưa và nhiệt độ
    days_open = (df["date"] - df["open_date"]).dt.days
    X_num = pd.DataFrame(index=df.index)
    X_num["grand_open"] = days_open.between(0, 2).astype(float)
    X_num["ramp_deficit"] = np.where(days_open >= 3, np.clip(63 - days_open, 0, 60) / 60.0, 0.0)
    X_num["uni_summer"] = ((st == "university") & (df["date"].dt.month.isin([7, 8]))).astype(float)
    X_num["global_trend"] = (df["date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25
    X_num["is_promo_global"] = df["is_promo"].astype(float)

    for s_type in ["mall", "office", "residential", "street", "university"]:
        m = (st == s_type).astype(float)
        X_num[f"promo_{s_type}"] = m * df["is_promo"].astype(float)
        X_num[f"rain_lin_{s_type}"] = m * (df["rain_mm"] / 50.0)
        X_num[f"temp_hn_{s_type}"] = m * (city == "Ha Noi").astype(float) * ((df["temp_c"] - 24.0) / 10.0)
        X_num[f"temp_hcm_{s_type}"] = m * (city == "Ho Chi Minh").astype(float) * ((df["temp_c"] - 28.0) / 10.0)

    X_sh = pd.concat([X_cat, X_num], axis=1)
    if shared_columns is not None:
        X_sh = X_sh.reindex(columns=shared_columns, fill_value=0.0)
    return X_sh


def build_gbdt_matrix(
    df: pd.DataFrame,
    p_stage1: np.ndarray,
    store_ints: pd.Series,
    gbdt_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Tạo ma trận đặc trưng cho bộ tăng cường HistGradientBoostingRegressor (Stage 2)."""
    gbdt_num = pd.DataFrame(
        {
            "store_int": df["store_id"].map(store_ints).values,
            "p_stage1": p_stage1,
            "dow": df["date"].dt.dayofweek.values,
            "is_promo": df["is_promo"].astype(float).values,
            "temp_c": df["temp_c"].values,
            "rain_mm": df["rain_mm"].values,
            "days_open": (df["date"] - df["open_date"]).dt.days.values,
            "seats": df["seats"].values,
            "area_m2": df["area_m2"].values,
            "has_parking": df["has_parking"].values,
        },
        index=df.index,
    )
    gbdt_cat = pd.get_dummies(df[["city", "store_type", "tet_phase"]], dtype=float)
    X_gbdt = pd.concat([gbdt_num, gbdt_cat], axis=1)
    if gbdt_columns is not None:
        X_gbdt = X_gbdt.reindex(columns=gbdt_columns, fill_value=0.0)
    return X_gbdt
