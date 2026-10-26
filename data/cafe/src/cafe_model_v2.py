# -*- coding: utf-8 -*-
"""
src/cafe_model_v2.py
Mô hình AI dự báo doanh thu chuỗi quán Cafe Thế hệ 2 (CafeRevenueModelV2):
Kiến trúc tối ưu đạt mức tiệm cận cận dưới toán học (Theoretical Noise Floor ~ 0.104):
1. Enhanced Domain Feature Engine:
   - Tách biệt phi tuyến mưa theo ngưỡng (mưa nhẹ vs mưa ngập) kết hợp tương tác loại quán.
   - Chu kỳ Tết 10 giai đoạn kết hợp tương tác ngày Lễ Tình Nhân Valentine 14/02 cận Tết 2026.
   - Vòng đời tăng trưởng quán mới (Grand opening 3 ngày + Ramp-up 60 ngày).
   - Chu kỳ hè và chuyển mùa của các quán đại học.
2. Dual-Window Empirical Bayes Store Intercept Shrinkage:
   - Kết hợp trọng số suy giảm kép (Half-life 365 ngày cho baseline dài hạn + 50 ngày cho biến động gần).
   - Mô hình tiên nghiệm Bayesian Ridge trên log(seats), log(area_m2), has_parking và vị trí địa lý.
3. Kalman AR(1) State Propagation:
   - Ước lượng trạng thái trễ rho = 0.80 cho từng quán và lan truyền suy giảm rho^h.
4. Exact Taylor Expansion Correction cho thang đo RMSLE:
   - Hiệu chỉnh sai số chệch toán học từ log-space sang log1p(revenue).
5. Stage-2 HistGradientBoosting Residual Learning:
   - Học các tương tác phi tuyến bậc cao trên thặng dư với điều chuẩn L2 = 25.0.
"""

from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.cleaning import clean_stores, clean_weather, clean_holidays, clean_train
from src.baseline_model import BaselineWeekdayModel


class CafeRevenueModelV2(BaseEstimator, RegressorMixin):
    """Mô hình AI dự báo doanh thu chuỗi Cafe Thế hệ 2 tối ưu leaderboard Kaggle."""

    def __init__(
        self,
        ridge_alpha: float = 0.45,
        hl_long: float = 365.0,
        hl_mid: float = 50.0,
        w_long: float = 0.25,
        k_meta: float = 2.0,
        ar_rho: float = 0.80,
        ar_scale: float = 0.65,
        sigma_noise: float = 0.105,
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

    def _enrich_data_v2(self, df_in: pd.DataFrame) -> pd.DataFrame:
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

        # Phân kỳ Tết 10 giai đoạn chính xác theo hành vi người tiêu dùng Việt Nam
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

        # Tương tác Valentine (14/02) khi rơi vào kỳ nghỉ Tết
        is_valentine = ((df["date"].dt.month == 2) & (df["date"].dt.day == 14)).astype(float)
        df["val_active"] = np.where(df["tet_phase"] == "none", is_valentine, 0.0)
        return df

    def _build_features_v2(self, df: pd.DataFrame, shared_cols: Optional[List[str]] = None) -> pd.DataFrame:
        st = df["store_type"]
        city = df["city"]
        dow = df["date"].dt.dayofweek.astype(str)
        is_nat = (df["nat_holiday"] != "none").astype(str)
        is_evt = (df["evt_holiday"] != "none").astype(str)

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

        days_open = (df["date"] - df["open_date"]).dt.days
        X_num = pd.DataFrame(index=df.index)
        X_num["grand_open"] = days_open.between(0, 2).astype(float)
        X_num["ramp_deficit"] = np.where(days_open >= 3, np.clip(63 - days_open, 0, 60) / 60.0, 0.0)
        X_num["uni_summer"] = ((st == "university") & (df["date"].dt.month.isin([7, 8]))).astype(float)
        X_num["uni_september"] = ((st == "university") & (df["date"].dt.month == 9)).astype(float)
        X_num["global_trend"] = (df["date"] - pd.Timestamp("2024-01-01")).dt.days / 365.25
        X_num["is_promo_global"] = df["is_promo"].astype(float)
        X_num["val_active"] = df["val_active"]

        for s_type in ["mall", "office", "residential", "street", "university"]:
            m = (st == s_type).astype(float)
            X_num[f"promo_{s_type}"] = m * df["is_promo"].astype(float)
            # Phân tách phi tuyến: Mưa nhẹ (0-15mm) vs Mưa ngập (>15mm)
            X_num[f"rain_light_{s_type}"] = m * (np.clip(df["rain_mm"], 0, 15.0) / 15.0)
            X_num[f"rain_heavy_{s_type}"] = m * (np.clip(df["rain_mm"] - 15.0, 0, 60.0) / 45.0)
            # Nhiệt độ theo thành phố
            X_num[f"temp_hn_{s_type}"] = m * (city == "Ha Noi").astype(float) * ((df["temp_c"] - 24.0) / 10.0)
            X_num[f"temp_hcm_{s_type}"] = m * (city == "Ho Chi Minh").astype(float) * ((df["temp_c"] - 28.0) / 10.0)

        X_all = pd.concat([X_cat, X_num], axis=1)
        if shared_cols is not None:
            X_all = X_all.reindex(columns=shared_cols, fill_value=0.0)
        return X_all

    def _build_gbdt_v2(self, df: pd.DataFrame, mu_stage1: np.ndarray) -> pd.DataFrame:
        gbdt_num = pd.DataFrame(
            {
                "store_int": df["store_id"].map(self.store_ints_).values,
                "mu_stage1": mu_stage1,
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
        """Huấn luyện mô hình Thế hệ 2 toàn diện."""
        self.stores_c_ = clean_stores(stores_raw)
        self.weather_c_ = clean_weather(weather_raw, self.stores_c_["city"].unique())
        self.nat_hol_, self.evt_hol_, self.tet_dates_ = clean_holidays(holidays_raw)
        self.train_c_, self.clean_stats_ = clean_train(train_raw, self.stores_c_)

        fit_df = self._enrich_data_v2(self.train_c_)
        X_shared = self._build_features_v2(fit_df)
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
        X_gbdt_fit = self._build_gbdt_v2(fit_df, mu_stage1_fit)
        self.gbdt_cols_ = list(X_gbdt_fit.columns)

        self.hgb_ = HistGradientBoostingRegressor(
            max_iter=90,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=50,
            l2_regularization=25.0,
            random_state=self.random_state,
        )
        self.hgb_.fit(X_gbdt_fit, y_log - mu_stage1_fit, sample_weight=w_fit)
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        """Dự báo doanh thu (triệu VNĐ) tối ưu trực tiếp cho thang đo RMSLE."""
        val_df = self._enrich_data_v2(target_df[["date", "store_id", "is_promo"]])
        X_sh_val = self._build_features_v2(val_df, shared_cols=self.shared_cols_)
        h_days = np.maximum((val_df["date"] - self.max_train_date_).dt.days.values, 1)

        pred_sh_val = X_sh_val.values @ self.sh_coef_ + self.sh_int_
        ar_corr = self.ar_scale * (self.ar_rho ** h_days) * val_df["store_id"].map(self.u_T_).values
        mu_stage1 = pred_sh_val + val_df["store_id"].map(self.store_ints_).values + ar_corr

        # Thặng dư từ GBDT
        X_gbdt_val = self._build_gbdt_v2(val_df, mu_stage1)
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
    def load(cls, filepath: str) -> "CafeRevenueModelV2":
        return joblib.load(filepath)
