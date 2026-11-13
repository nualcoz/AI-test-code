# -*- coding: utf-8 -*-
"""
src/cafe_model_v3.py
Mô hình AI dự báo doanh thu chuỗi quán Cafe Thế hệ 3 (CafeRevenueModelV3):
Được tối ưu hóa toàn diện khi tích hợp dữ liệu kiểm thử thực tế test.csv:
1. Enhanced Domain Feature Engine v3:
   - Tận dụng 100% nhãn khuyến mãi chính xác (is_promo = 0/1) từ test.csv.
   - Tương tác đa chiều: Khuyến mãi x Loại quán, Khuyến mãi x Cuối tuần (Promo Weekend Effect).
   - Tương tác Lễ Tình Nhân Valentine 14/02/2026 (27 tháng Chạp Bính Ngọ) & Quốc tế Phụ nữ 08/03.
   - Phân kỳ Tết 10 giai đoạn chuẩn xác theo văn hóa tiêu dùng và lịch âm Việt Nam.
   - Phân tách phi tuyến tác động thời tiết: Mưa nhẹ (0-15mm) vs Mưa ngập (>15mm), nhiệt độ theo vùng.
   - Vòng đời cửa hàng mới (Grand Opening 3 ngày + Ramp-up 60 ngày).
2. Dual-Window Empirical Bayes Store Intercepts:
   - Kết hợp trọng số suy giảm kép (Half-life 365 ngày cho baseline dài hạn + 50 ngày cho biến động gần).
   - Bayesian Ridge prior shrinkage dựa trên log(seats), log(area_m2), has_parking và vị trí địa lý.
3. Kalman AR(1) State Propagation:
   - Ước lượng trạng thái trôi dạt ngắn hạn rho = 0.80 cho từng quán và lan truyền suy giảm rho^h sang Q1/2026.
4. Stage-2 Non-linear GBDT Residual Learning:
   - Học bù phần dư phi tuyến bằng HistGradientBoostingRegressor với điều chuẩn L2 cao.
5. Exact Second-Order Taylor Correction:
   - Tối ưu hóa trực tiếp cho hàm mục tiêu RMSLE trên log(1 + revenue).
"""

from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.cleaning import clean_stores, clean_weather, clean_holidays, clean_train, clean_test
from src.config import EVAL_QUARTERS


class CafeRevenueModelV3(BaseEstimator, RegressorMixin):
    """Mô hình AI dự báo doanh thu chuỗi Cafe Thế hệ 3 tối ưu leaderboard Kaggle với dữ liệu test thực tế."""

    def __init__(
        self,
        ridge_alpha: float = 0.40,
        hl_long: float = 365.0,
        hl_mid: float = 50.0,
        w_long: float = 0.25,
        k_meta: float = 2.0,
        ar_rho: float = 0.80,
        ar_scale: float = 0.65,
        sigma_noise: float = 0.104,
        gbdt_weight: float = 0.08,
        random_state: int = 42,
    ):
        self.ridge_alpha = ridge_alpha
        self.hl_long = hl_long
        self.hl_mid = hl_mid
        self.w_long = w_long
        self.k_meta = k_meta
        self.ar_rho = ar_rho
        self.ar_scale = ar_scale
        self.sigma_noise = sigma_noise
        self.gbdt_weight = gbdt_weight
        self.random_state = random_state

    def _enrich_data(self, df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(self.stores_c_, on="store_id", how="left").merge(
            self.weather_c_, on=["date", "city"], how="left"
        )
        df = df.merge(self.nat_hol_, on="date", how="left").merge(self.evt_hol_, on="date", how="left")
        df["nat_holiday"] = df["nat_holiday"].fillna("none")
        df["evt_holiday"] = df["evt_holiday"].fillna("none")
        df["tet_date"] = df["date"].dt.year.map(self.tet_dates_)
        df["tet_offset"] = (df["date"] - df["tet_date"]).dt.days

        # Phân kỳ Tết 10 giai đoạn
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

        # Ngày lễ đặc biệt
        df["is_valentine"] = ((df["date"].dt.month == 2) & (df["date"].dt.day == 14)).astype(float)
        df["is_womens_day"] = ((df["date"].dt.month == 3) & (df["date"].dt.day == 8)).astype(float)
        df["is_new_year"] = ((df["date"].dt.month == 1) & (df["date"].dt.day == 1)).astype(float)
        return df

    def _build_features(self, df: pd.DataFrame, shared_cols: Optional[List[str]] = None) -> pd.DataFrame:
        st = df["store_type"]
        city = df["city"]
        dow = df["date"].dt.dayofweek.astype(str)
        is_weekend = df["date"].dt.dayofweek.isin([5, 6]).astype(float)
        is_nat = (df["nat_holiday"] != "none").astype(str)
        is_evt = (df["evt_holiday"] != "none").astype(str)

        # Categorical dummies
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

        # Numerical & Interaction features
        days_open = (df["date"] - df["open_date"]).dt.days
        X_num = pd.DataFrame(index=df.index)
        X_num["grand_open"] = days_open.between(0, 2).astype(float)
        X_num["ramp_deficit"] = np.where(days_open >= 3, np.clip(63 - days_open, 0, 60) / 60.0, 0.0)
        X_num["uni_summer"] = ((st == "university") & (df["date"].dt.month.isin([7, 8]))).astype(float)
        X_num["uni_september"] = ((st == "university") & (df["date"].dt.month == 9)).astype(float)
        X_num["global_trend"] = (df["date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25
        
        # Exact promotional features from test.csv / train.csv
        promo_flt = df["is_promo"].astype(float)
        X_num["is_promo_global"] = promo_flt
        X_num["promo_weekend"] = promo_flt * is_weekend
        
        # Special Day features
        X_num["is_valentine"] = df["is_valentine"]
        X_num["is_womens_day"] = df["is_womens_day"]
        X_num["is_new_year"] = df["is_new_year"]

        for s_type in ["mall", "office", "residential", "street", "university"]:
            m = (st == s_type).astype(float)
            X_num[f"promo_{s_type}"] = m * promo_flt
            # Weather effects: non-linear light rain vs heavy rain
            X_num[f"rain_light_{s_type}"] = m * (np.clip(df["rain_mm"], 0, 15.0) / 15.0)
            X_num[f"rain_heavy_{s_type}"] = m * (np.clip(df["rain_mm"] - 15.0, 0, 60.0) / 45.0)
            # Temperature anomalies by city
            X_num[f"temp_hn_{s_type}"] = m * (city == "Ha Noi").astype(float) * ((df["temp_c"] - 24.0) / 10.0)
            X_num[f"temp_hcm_{s_type}"] = m * (city == "Ho Chi Minh").astype(float) * ((df["temp_c"] - 28.0) / 10.0)
            X_num[f"temp_dn_{s_type}"] = m * (city == "Da Nang").astype(float) * ((df["temp_c"] - 26.0) / 10.0)

        X_all = pd.concat([X_cat, X_num], axis=1)
        if shared_cols is not None:
            X_all = X_all.reindex(columns=shared_cols, fill_value=0.0)
        return X_all

    def _build_gbdt(self, df: pd.DataFrame, mu_stage1: np.ndarray) -> pd.DataFrame:
        gbdt_num = pd.DataFrame(
            {
                "store_int": df["store_id"].map(self.store_ints_).values,
                "mu_stage1": mu_stage1,
                "dow": df["date"].dt.dayofweek.values,
                "is_weekend": df["date"].dt.dayofweek.isin([5, 6]).astype(float).values,
                "is_promo": df["is_promo"].astype(float).values,
                "temp_c": df["temp_c"].values,
                "rain_mm": df["rain_mm"].values,
                "days_open": (df["date"] - df["open_date"]).dt.days.values,
                "seats": df["seats"].values,
                "area_m2": df["area_m2"].values,
                "has_parking": df["has_parking"].values,
                "is_valentine": df["is_valentine"].values,
                "is_womens_day": df["is_womens_day"].values,
            },
            index=df.index,
        )
        gbdt_cat = pd.get_dummies(df[["city", "store_type", "tet_phase"]], dtype=float)
        X_gbdt = pd.concat([gbdt_num, gbdt_cat], axis=1)
        if hasattr(self, "gbdt_cols_"):
            X_gbdt = X_gbdt.reindex(columns=self.gbdt_cols_, fill_value=0.0)
        return X_gbdt

    def fit(
        self,
        train_raw: pd.DataFrame,
        stores_raw: pd.DataFrame,
        weather_raw: pd.DataFrame,
        holidays_raw: pd.DataFrame,
    ):
        """Huấn luyện mô hình Thế hệ 3 toàn diện trên dữ liệu lịch sử."""
        self.stores_c_ = clean_stores(stores_raw)
        self.weather_c_ = clean_weather(weather_raw, self.stores_c_["city"].unique())
        self.nat_hol_, self.evt_hol_, self.tet_dates_ = clean_holidays(holidays_raw)
        self.train_c_, self.clean_stats_ = clean_train(train_raw, self.stores_c_)

        fit_df = self._enrich_data(self.train_c_)
        X_shared = self._build_features(fit_df)
        self.shared_cols_ = list(X_shared.columns)

        X_store = pd.get_dummies(fit_df["store_id"], prefix="sid", dtype=float)
        self.store_cols_ = list(X_store.columns)
        X_all = pd.concat([X_store, X_shared], axis=1)

        w_fit = np.where(fit_df["has_exact_weather"].values, 1.0, 0.15)
        self.max_train_date_ = fit_df["date"].max()
        days_ago = (self.max_train_date_ - fit_df["date"]).dt.days.values

        # Stage 1: Ridge trên log(revenue)
        y_log = np.log(self.train_c_["revenue"].values)
        ridge_model = Ridge(alpha=self.ridge_alpha).fit(X_all, y_log, sample_weight=w_fit)
        self.sh_coef_ = ridge_model.coef_[len(self.store_cols_) :]
        self.sh_int_ = float(ridge_model.intercept_)

        pred_sh_fit = X_shared.values @ self.sh_coef_ + self.sh_int_
        res_fit = y_log - pred_sh_fit

        fit_df["raw_res"] = res_fit
        cd_shock = fit_df.groupby(["city", "date"])["raw_res"].transform("mean")
        clean_res = np.where(fit_df["has_exact_weather"].values, res_fit, res_fit - 0.85 * cd_shock.values)

        # Dual-Window Exponential Weighting
        w_l = np.exp(-np.log(2) * days_ago / self.hl_long) * w_fit
        w_m = np.exp(-np.log(2) * days_ago / self.hl_mid) * w_fit
        tmp = pd.DataFrame(
            {
                "store_id": fit_df["store_id"].values,
                "wl_res": clean_res * w_l,
                "wl": w_l,
                "wm_res": clean_res * w_m,
                "wm": w_m,
            }
        )
        g = tmp.groupby("store_id").sum()
        int_long = g["wl_res"] / g["wl"]
        int_mid = g["wm_res"] / g["wm"]
        emp_store_int = self.w_long * int_long + (1.0 - self.w_long) * int_mid

        counts = fit_df["store_id"].value_counts()
        seen = counts[counts > 0].index.tolist()
        w_meta = np.minimum(counts.loc[seen].values, 200.0) / 200.0

        # Bayesian Prior Regression từ Store Metadata
        meta_X = pd.get_dummies(self.stores_c_[["city", "store_type"]], dtype=float)
        meta_X["log_seats"] = (np.log(self.stores_c_["seats"]) - np.log(self.stores_c_["seats"]).mean()).values
        meta_X["log_area"] = (np.log(self.stores_c_["area_m2"]) - np.log(self.stores_c_["area_m2"]).mean()).values
        meta_X["has_parking"] = self.stores_c_["has_parking"].astype(float).values
        meta_X.index = self.stores_c_["store_id"]
        m_meta = Ridge(alpha=0.03).fit(meta_X.loc[seen], int_long.loc[seen], sample_weight=w_meta)
        pred_meta = pd.Series(m_meta.predict(meta_X), index=self.stores_c_["store_id"])

        # Co cụm James-Stein / Empirical Bayes
        self.store_ints_ = pd.Series(
            {
                sid: (counts.get(sid, 0) / (counts.get(sid, 0) + self.k_meta)) * emp_store_int.get(sid, 0.0)
                + (self.k_meta / (counts.get(sid, 0) + self.k_meta)) * pred_meta[sid]
                if counts.get(sid, 0) > 0
                else pred_meta[sid]
                for sid in self.stores_c_["store_id"]
            }
        )

        # AR(1) State Estimation
        dev_res = clean_res - fit_df["store_id"].map(self.store_ints_).values
        w_ar = (self.ar_rho ** days_ago) * (days_ago <= 21).astype(float) * w_fit
        tmp_ar = pd.DataFrame({"store_id": fit_df["store_id"].values, "war_res": dev_res * w_ar, "war": w_ar})
        g_ar = tmp_ar.groupby("store_id").sum()
        self.u_T_ = (g_ar["war_res"] / (g_ar["war"] + 0.8)).reindex(self.stores_c_["store_id"]).fillna(0.0)

        # Stage 2: HistGradientBoosting trên thặng dư
        mu_stage1_fit = pred_sh_fit + fit_df["store_id"].map(self.store_ints_).values
        X_gbdt_fit = self._build_gbdt(fit_df, mu_stage1_fit)
        self.gbdt_cols_ = list(X_gbdt_fit.columns)

        self.hgb_ = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=45,
            l2_regularization=20.0,
            random_state=self.random_state,
        )
        self.hgb_.fit(X_gbdt_fit, y_log - mu_stage1_fit, sample_weight=w_fit)
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        """Dự báo doanh thu (triệu VNĐ) tối ưu trực tiếp cho thang đo RMSLE."""
        val_df = self._enrich_data(target_df[["date", "store_id", "is_promo"]])
        X_sh_val = self._build_features(val_df, shared_cols=self.shared_cols_)
        h_days = np.maximum((val_df["date"] - self.max_train_date_).dt.days.values, 1)

        pred_sh_val = X_sh_val.values @ self.sh_coef_ + self.sh_int_
        ar_corr = self.ar_scale * (self.ar_rho ** h_days) * val_df["store_id"].map(self.u_T_).values
        mu_stage1 = pred_sh_val + val_df["store_id"].map(self.store_ints_).values + ar_corr

        # Thặng dư từ GBDT Stage 2
        X_gbdt_val = self._build_gbdt(val_df, mu_stage1)
        res_gbdt = self.hgb_.predict(X_gbdt_val)

        mu_final = mu_stage1 + self.gbdt_weight * res_gbdt

        # Exact Second-order Taylor Expansion Bias Correction cho RMSLE
        sigma2 = self.sigma_noise ** 2
        ep = np.exp(mu_final)
        u_star = np.log1p(ep) + 0.5 * sigma2 * ep / ((1.0 + ep) ** 2)
        pred_rev = np.expm1(u_star)
        return np.clip(pred_rev, 0.0, None)

    def save(self, filepath: str) -> str:
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CafeRevenueModelV3":
        return joblib.load(filepath)
