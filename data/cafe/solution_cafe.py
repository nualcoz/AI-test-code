# -*- coding: utf-8 -*-
"""Bài giải dự báo doanh thu chuỗi Cafe (Q1/2026) - Metric: RMSLE"""

import os, glob
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

# Tự động tìm thư mục chứa dữ liệu
hits = (glob.glob("/kaggle/input/**/stores.csv", recursive=True)
        + glob.glob("stores.csv")
        + glob.glob("data/stores.csv"))
DATA = (os.path.dirname(hits[0]) or ".") if hits else "."
print("Thư mục dữ liệu:", DATA)

train = pd.read_csv(f"{DATA}/train.csv", parse_dates=["date"])
stores = pd.read_csv(f"{DATA}/stores.csv")
weather = pd.read_csv(f"{DATA}/weather.csv", parse_dates=["date"])
holidays = pd.read_csv(f"{DATA}/holidays.csv", parse_dates=["date"])
sample_sub = pd.read_csv(f"{DATA}/sample_submission.csv")

if os.path.exists(f"{DATA}/test.csv"):
    test = pd.read_csv(f"{DATA}/test.csv", parse_dates=["date"])
    print("Đã tải test.csv trực tiếp từ thư mục dữ liệu.")
else:
    # Khôi phục lưới 3,869 dòng (id, date, store_id) của Q1/2026 khớp với sample_submission.csv
    st_tmp = stores.assign(st=stores.store_type.str.lower().str.strip())
    active = st_tmp[st_tmp.store_id != "S02"][["store_id", "st"]]
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2026-01-01", "2026-03-31"), active.store_id],
        names=["date", "store_id"]
    ).to_frame().reset_index(drop=True)
    grid = grid.merge(active, on="store_id").sort_values(["date", "store_id"]).reset_index(drop=True)
    tet_off = (grid.date - pd.Timestamp("2026-02-17")).dt.days
    closed = ((grid.st == "office") & tet_off.between(-2, 2)) | ((grid.st == "university") & tet_off.between(-2, 3))
    test = grid[~closed][["date", "store_id"]].reset_index(drop=True)
    test.insert(0, "id", sample_sub.id.values)
    test["is_promo"] = 0
    print("Đã dựng lưới test Q1/2026 (3,869 dòng) khớp với sample_submission.csv.")

for name, df in [("train", train), ("test", test), ("stores", stores), ("weather", weather), ("holidays", holidays)]:
    if "date" in df:
        print(f"{name:10s}: {str(df.shape):12s} | {df.date.min().date()} -> {df.date.max().date()}")
    else:
        print(f"{name:10s}: {str(df.shape):12s}")

def clean_stores(stores_raw):
    s = stores_raw.copy()
    s["store_type"] = s["store_type"].str.lower().str.strip()
    s["district"] = s["district"].str.strip().replace({
        "Q.10": "District 10", "Quan 3": "District 3", "Quan 7": "District 7"
    })
    s["open_date"] = pd.to_datetime(s["open_date"], format="mixed", dayfirst=True)
    ratio_by_type = (s["area_m2"] / s["seats"]).groupby(s["store_type"]).median()
    for idx, row in s[s["area_m2"].isna()].iterrows():
        s.loc[idx, "area_m2"] = round(row["seats"] * ratio_by_type[row["store_type"]], 1)
    return s

def clean_weather(weather_raw, cities):
    w = weather_raw.copy()
    f_mask = w["temp_c"] > 45.0
    w.loc[f_mask, "temp_c"] = (w.loc[f_mask, "temp_c"] - 32.0) * 5.0 / 9.0
    grid = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", "2026-03-31"), sorted(cities)],
        names=["date", "city"]
    ).to_frame().reset_index(drop=True)
    w = grid.merge(w, on=["date", "city"], how="left").sort_values(["city", "date"]).reset_index(drop=True)
    w["temp_c"] = w.groupby("city")["temp_c"].transform(lambda x: x.interpolate().bfill().ffill())
    w["rain_mm"] = w["rain_mm"].fillna(0.0)
    return w

def clean_holidays(holidays_raw):
    non_tet_nat = holidays_raw[(holidays_raw["type"] == "national_holiday") & (~holidays_raw["holiday_name"].str.contains("Tet"))]
    nat_hol = non_tet_nat.groupby("date")["holiday_name"].first().rename("nat_holiday")
    evt_hol = holidays_raw[(holidays_raw["type"] == "event") & (~holidays_raw["holiday_name"].str.contains("Tet"))].groupby("date")["holiday_name"].first().rename("evt_holiday")
    tet_rows = holidays_raw[holidays_raw["holiday_name"] == "Lunar New Year (Tet)"]
    tet_dates = dict(zip(tet_rows["date"].dt.year, tet_rows["date"]))
    return nat_hol, evt_hol, tet_dates

def clean_train(df_in, stores_clean):
    df = df_in.drop_duplicates(subset=["date", "store_id"]).copy()
    df = df.merge(stores_clean[["store_id", "open_date"]], on="store_id", how="left")
    df["dow"] = df["date"].dt.dayofweek
    df["days_since_open"] = (df["date"] - df["open_date"]).dt.days
    
    n_1000x = int((df["revenue"] > 1000).sum())
    df.loc[df["revenue"] > 1000, "revenue"] /= 1000.0
    
    med_dow = df.groupby(["store_id", "dow"])["revenue"].transform("median")
    ratio_dow = df["revenue"] / med_dow
    mask_10x = (df["days_since_open"] > 14) & (ratio_dow > 3.0)
    mask_01x = ratio_dow < 0.20
    df.loc[mask_10x, "revenue"] /= 10.0
    df.loc[mask_01x, "revenue"] *= 10.0
    return df[["date", "store_id", "is_promo", "revenue"]], (len(df_in) - len(df), n_1000x, int(mask_10x.sum()), int(mask_01x.sum()))

stores_c = clean_stores(stores)
weather_c = clean_weather(weather, stores_c["city"].unique())
nat_hol, evt_hol, tet_dates = clean_holidays(holidays)
train_c, stats = clean_train(train, stores_c)

print(f"Đã làm sạch train: {train_c.shape} (Xóa {stats[0]} dòng trùng, sửa {stats[1]} dòng 1000x, {stats[2]} dòng 10x, {stats[3]} dòng 0.1x)")
print(train_c["revenue"].describe().round(2))

def rmsle(y, p):
    return np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(y)) ** 2))

def enrich_data(df_in, stores_df, weather_df, nat_hol_s, evt_hol_s, tet_dates_map):
    df = df_in.merge(stores_df, on="store_id", how="left").merge(weather_df, on=["date", "city"], how="left")
    df = df.merge(nat_hol_s, on="date", how="left").merge(evt_hol_s, on="date", how="left")
    df["nat_holiday"] = df["nat_holiday"].fillna("none")
    df["evt_holiday"] = df["evt_holiday"].fillna("none")
    df["tet_date"] = df["date"].dt.year.map(tet_dates_map)
    df["tet_offset"] = (df["date"] - df["tet_date"]).dt.days
    
    df["tet_phase"] = "none"
    df.loc[df["tet_offset"].between(-14, -13), "tet_phase"] = "pre_tet_14_13"
    df.loc[df["tet_offset"].between(-12, -8),  "tet_phase"] = "pre_tet_12_8"
    df.loc[df["tet_offset"] == -7,             "tet_phase"] = "tet_23_ong_tao"
    df.loc[df["tet_offset"].between(-6, -3),   "tet_phase"] = "pre_tet_6_3"
    df.loc[df["tet_offset"].between(-2, 1),    "tet_phase"] = "tet_core_m2_p1"
    df.loc[df["tet_offset"] == 2,              "tet_phase"] = "tet_p2"
    df.loc[df["tet_offset"].between(3, 7),     "tet_phase"] = "post_tet_3_7"
    df.loc[df["tet_offset"].between(8, 9),     "tet_phase"] = "post_tet_8_9"
    df.loc[df["tet_offset"].between(10, 12),   "tet_phase"] = "post_tet_10_12"
    return df

def build_shared_matrix(df):
    st = df["store_type"]
    city = df["city"]
    dow = df["date"].dt.dayofweek.astype(str)
    cat_df = pd.DataFrame({
        "st_city_dow": st + "_" + city + "_" + dow,
        "st_promo":    st + "_" + df["is_promo"].astype(str),
        "st_nat_hol":  st + "_" + df["nat_holiday"],
        "st_evt_hol":  st + "_" + df["evt_holiday"],
        "st_tet":      st + "_" + df["tet_phase"],
    }, index=df.index)
    X_cat = pd.get_dummies(cat_df, dtype=float)
    
    days_open = (df["date"] - df["open_date"]).dt.days
    X_num = pd.DataFrame(index=df.index)
    X_num["open_d0"] = (days_open == 0).astype(float)
    X_num["open_d1"] = (days_open == 1).astype(float)
    X_num["open_d2"] = (days_open == 2).astype(float)
    X_num["ramp_deficit"] = np.where(days_open >= 3, np.clip(63 - days_open, 0, 60) / 60.0, 0.0)
    X_num["uni_summer"] = ((st == "university") & (df["date"].dt.month.isin([7, 8]))).astype(float)
    X_num["global_trend"] = (df["date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25
    
    for s_type in ["mall", "office", "residential", "street", "university"]:
        m = (st == s_type).astype(float)
        X_num[f"rain_log_{s_type}"]    = m * np.log1p(df["rain_mm"])
        X_num[f"rain_lin_{s_type}"]    = m * (df["rain_mm"] / 50.0)
        X_num[f"heavy_rain_{s_type}"]  = m * (df["rain_mm"] > 10.0).astype(float)
        X_num[f"vheavy_rain_{s_type}"] = m * (df["rain_mm"] > 30.0).astype(float)
        X_num[f"temp_hn_{s_type}"]     = m * (city == "Ha Noi").astype(float) * ((df["temp_c"] - 24.0) / 10.0)
        X_num[f"temp_hcm_{s_type}"]    = m * (city == "Ho Chi Minh").astype(float) * ((df["temp_c"] - 28.0) / 10.0)
    return pd.concat([X_cat, X_num], axis=1)

def fit_stage1(X_all, X_shared, X_store_dummies, fit_df, val_df, y_fit, alpha=1.0):
    n_fit = len(fit_df)
    model = Ridge(alpha=alpha).fit(X_all.iloc[:n_fit], y_fit)
        
    pred_shared_fit = X_shared.iloc[:n_fit].values @ model.coef_[X_store_dummies.shape[1]:] + model.intercept_
    res_fit = y_fit - pred_shared_fit
    
    days_ago = (fit_df["date"].max() - fit_df["date"]).dt.days.values
    weights = np.exp(-np.log(2) * days_ago / 120.0)
    tmp = pd.DataFrame({"store_id": fit_df["store_id"].values, "w_res": res_fit * weights, "w": weights})
    g = tmp.groupby("store_id").sum()
    emp_store_int = g["w_res"] / g["w"]
    
    counts = fit_df["store_id"].value_counts()
    seen_stores = counts[counts > 0].index.tolist()
    meta_X = pd.get_dummies(stores_c[["city", "store_type", "has_parking"]], dtype=float)
    meta_X["log_seats"] = np.log(stores_c["seats"])
    meta_X["log_area"] = np.log(stores_c["area_m2"])
    meta_X.index = stores_c["store_id"]
    meta_model = Ridge(alpha=0.5).fit(
        meta_X.loc[seen_stores], emp_store_int.loc[seen_stores], sample_weight=np.sqrt(counts.loc[seen_stores])
    )
    pred_meta_int = pd.Series(meta_model.predict(meta_X), index=stores_c["store_id"])
    
    final_store_int = pd.Series({
        sid: (counts.get(sid, 0) / (counts.get(sid, 0) + 5.0)) * emp_store_int.get(sid, 0.0)
             + (5.0 / (counts.get(sid, 0) + 5.0)) * pred_meta_int[sid]
        if counts.get(sid, 0) > 0 else pred_meta_int[sid]
        for sid in stores_c["store_id"]
    })
    
    pred_shared_val = X_shared.iloc[n_fit:].values @ model.coef_[X_store_dummies.shape[1]:] + model.intercept_
    p_fit = pred_shared_fit + fit_df["store_id"].map(final_store_int).values
    p_val = pred_shared_val + val_df["store_id"].map(final_store_int).values
    return p_fit, p_val, final_store_int

def predict_cafe_revenue(history_raw, target_df):
    hist_c, _ = clean_train(history_raw, stores_c)
    comb = enrich_data(
        pd.concat([hist_c, target_df[["date", "store_id", "is_promo"]]], ignore_index=True),
        stores_c, weather_c, nat_hol, evt_hol, tet_dates
    )
    X_shared = build_shared_matrix(comb)
    X_store_dummies = pd.get_dummies(comb["store_id"], prefix="sid", dtype=float)
    X_all = pd.concat([X_store_dummies, X_shared], axis=1)
    
    n_fit = len(hist_c)
    fit_df, val_df = comb.iloc[:n_fit], comb.iloc[n_fit:]
    
    # Model 1: Ridge trên log(1 + revenue)
    y_log1p = np.log1p(hist_c["revenue"].values)
    p1_fit, p1_val, store_int_1 = fit_stage1(X_all, X_shared, X_store_dummies, fit_df, val_df, y_log1p, alpha=1.0)
    
    # Model 2: Ridge trên log(revenue), quy đổi về không gian log(1 + revenue)
    y_log = np.log(hist_c["revenue"].values)
    p2_fit_log, p2_val_log, _ = fit_stage1(X_all, X_shared, X_store_dummies, fit_df, val_df, y_log, alpha=1.0)
    p2_fit = np.log1p(np.exp(p2_fit_log))
    p2_val = np.log1p(np.exp(p2_val_log))
    
    p_base_fit = 0.55 * p1_fit + 0.45 * p2_fit
    p_base_val = 0.55 * p1_val + 0.45 * p2_val
    
    # Điều chỉnh thứ trong tuần đặc thù của từng cửa hàng (có co cụm)
    dow_res = pd.DataFrame({"store_id": fit_df["store_id"].values, "dow": fit_df["date"].dt.dayofweek.values, "r": y_log1p - p_base_fit})
    dow_g = dow_res.groupby(["store_id", "dow"])["r"].agg(["count", "mean"])
    dow_g["shrunk"] = (dow_g["count"] / (dow_g["count"] + 25.0)) * dow_g["mean"]
    p_stage1_fit = p_base_fit + pd.Series(list(zip(fit_df["store_id"], fit_df["date"].dt.dayofweek))).map(dow_g["shrunk"]).fillna(0.0).values
    p_stage1_val = p_base_val + pd.Series(list(zip(val_df["store_id"], val_df["date"].dt.dayofweek))).map(dow_g["shrunk"]).fillna(0.0).values
    
    # Model 4: HistGradientBoostingRegressor học phần dư phi tuyến
    gbdt_num = pd.DataFrame({
        "store_int":   comb["store_id"].map(store_int_1).values,
        "p_stage1":    np.concatenate([p_stage1_fit, p_stage1_val]),
        "dow":         comb["date"].dt.dayofweek.values,
        "month":       comb["date"].dt.month.values,
        "dayofyear":   comb["date"].dt.dayofyear.values,
        "is_promo":    comb["is_promo"].values,
        "temp_c":      comb["temp_c"].values,
        "rain_mm":     comb["rain_mm"].values,
        "days_open":   (comb["date"] - comb["open_date"]).dt.days.values,
        "tet_offset":  comb["tet_offset"].fillna(999).values,
        "seats":       comb["seats"].values,
        "area_m2":     comb["area_m2"].values,
        "has_parking": comb["has_parking"].values,
        "t_days":      (comb["date"] - pd.Timestamp("2024-01-01")).dt.days.values,
    })
    gbdt_cat = pd.get_dummies(comb[["city", "store_type", "nat_holiday", "evt_holiday", "tet_phase"]], dtype=float).reset_index(drop=True)
    X_gbdt = pd.concat([gbdt_num, gbdt_cat], axis=1)
    
    hgb = HistGradientBoostingRegressor(
        max_iter=120, learning_rate=0.03, max_depth=4, min_samples_leaf=40, l2_regularization=10.0, random_state=42
    )
    hgb.fit(X_gbdt.iloc[:n_fit], y_log1p - p_stage1_fit)
    res_gbdt = hgb.predict(X_gbdt.iloc[n_fit:])
    
    return np.expm1(p_stage1_val + 0.30 * res_gbdt)

def weekday_median_baseline(history, target):
    h = history.assign(dow=history.date.dt.dayofweek)
    m = h.groupby(["store_id", "dow"]).revenue.median()
    keys = list(zip(target.store_id, target.date.dt.dayofweek))
    p = pd.Series(keys).map(m)
    store_med = target.store_id.map(h.groupby("store_id").revenue.median())
    p = p.fillna(store_med.reset_index(drop=True))
    return p.fillna(h.revenue.median()).values

# Lát cắt 1: Q4/2025 (Oct-Dec 2025)
fit_q4 = train[train.date < "2025-10-01"]
val_q4_raw = train[train.date >= "2025-10-01"]
val_q4_clean = train_c[train_c.date >= "2025-10-01"]

pred_q4_raw = predict_cafe_revenue(fit_q4, val_q4_raw)
pred_q4_clean = predict_cafe_revenue(fit_q4, val_q4_clean)

# Lát cắt 2: Q1/2025 (Jan-Mar 2025)
fit_q1 = train[train.date < "2025-01-01"]
val_q1_clean = train_c[(train_c.date >= "2025-01-01") & (train_c.date <= "2025-03-31")]
pred_q1_clean = predict_cafe_revenue(fit_q1, val_q1_clean)

print("=== KẾT QUẢ ĐÁNH GIÁ RMSLE TRÊN TẬP VALIDATION ===")
print(f"1. Q4/2025 (Raw Val)   | Baseline: {rmsle(val_q4_raw.revenue.values, weekday_median_baseline(fit_q4, val_q4_raw)):.4f}  -->  Ensemble Model: {rmsle(val_q4_raw.revenue.values, pred_q4_raw):.4f}")
print(f"2. Q4/2025 (Clean Val) | Baseline: {rmsle(val_q4_clean.revenue.values, weekday_median_baseline(fit_q4, val_q4_clean)):.4f}  -->  Ensemble Model: {rmsle(val_q4_clean.revenue.values, pred_q4_clean):.4f}")
print(f"3. Q1/2025 (Clean Val) | Baseline: {rmsle(val_q1_clean.revenue.values, weekday_median_baseline(fit_q1, val_q1_clean)):.4f}  -->  Ensemble Model: {rmsle(val_q1_clean.revenue.values, pred_q1_clean):.4f}")

test_preds = np.round(predict_cafe_revenue(train, test), 3)
sub = pd.DataFrame({
    "id": test["id"].values,
    "revenue": np.clip(test_preds, 0.0, None)
})
sub.to_csv("submission.csv", index=False)

# Kiểm tra tính hợp lệ đối chiếu với sample_submission.csv
assert list(sub.columns) == ["id", "revenue"], "Sai tên cột!"
assert len(sub) == len(sample_sub), f"Sai số dòng: {len(sub)} != {len(sample_sub)}"
assert (sub["id"].values == sample_sub["id"].values).all(), "Cột id không khớp với sample_submission.csv!"
assert sub["revenue"].isna().sum() == 0, "Có giá trị NaN trong dự báo!"
assert (sub["revenue"] >= 0).all(), "Có giá trị âm trong dự báo!"

print(f"Đã xuất thành công submission.csv ({sub.shape[0]} dòng, khớp 100% id với sample_submission.csv)!")
print("\nThống kê phân phối doanh thu dự báo (Q1/2026):")
print(sub["revenue"].describe().round(2))
sub.head(10)
