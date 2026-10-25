"""
src/cafe_model.py
Mô hình AI dự báo doanh thu chuỗi quán Cafe (CafeRevenueModel):
- Kiến trúc 2 tầng (Hierarchical / Multi-stage) tối ưu thang đo RMSLE.
- Stage 1: Ridge Regression phân rã hiệu ứng chung (Khuyến mãi, Thời tiết, Ngày trong tuần, Lễ Tết, Khai trương, Mùa hè đại học, Xu hướng)
  và hiệu ứng quán cụ thể (Store-specific base intercept) với suy giảm hàm mũ theo thời gian + nội suy Empirical Bayes cho quán mới.
- Khắc phục hiện tượng trễ tự tương quan AR(1) nhiễu ngắn hạn bằng bước lọc Kalman.
- Stage 2: Tăng cường phi tuyến bằng HistGradientBoostingRegressor trên thặng dư.
"""

from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from src.cleaning import clean_stores, clean_weather, clean_holidays, clean_train
from src.features import enrich_data, build_shared_matrix, build_gbdt_matrix
from src.baseline_model import BaselineWeekdayModel


class CafeRevenueModel(BaseEstimator, RegressorMixin):
    """
    Mô hình phát triển trực tiếp từ Baseline (store_id, weekday):
      Dự báo = Baseline(store_id, dow) [đã làm sạch & nội suy quán mới]
               * Hệ số điều chỉnh (Khuyến mãi, Thời tiết, Lễ/Tết, Khai trương, AR(1), GBDT)
    """

    def __init__(
        self,
        ridge_alpha: float = 0.5,
        hl_long: float = 365.0,
        hl_mid: float = 50.0,
        w_long: float = 0.25,
        k_meta: float = 2.0,
        ar_rho: float = 0.80,
        ar_scale: float = 0.65,
        w_log1p: float = 0.25,
        gbdt_weight: float = 0.10,
        random_state: int = 42,
    ):
        self.ridge_alpha = ridge_alpha
        self.hl_long = hl_long
        self.hl_mid = hl_mid
        self.w_long = w_long
        self.k_meta = k_meta
        self.ar_rho = ar_rho
        self.ar_scale = ar_scale
        self.w_log1p = w_log1p
        self.gbdt_weight = gbdt_weight
        self.random_state = random_state

    def fit(
        self,
        train_raw: pd.DataFrame,
        stores_raw: pd.DataFrame,
        weather_raw: pd.DataFrame,
        holidays_raw: pd.DataFrame,
    ):
        """Huấn luyện mô hình từ bảng lịch sử train_raw và các bảng phụ trợ."""
        self.stores_c_ = clean_stores(stores_raw)
        self.weather_c_ = clean_weather(weather_raw, self.stores_c_["city"].unique())
        self.nat_hol_, self.evt_hol_, self.tet_dates_ = clean_holidays(holidays_raw)
        self.train_c_, self.clean_stats_ = clean_train(train_raw, self.stores_c_)

        # Lưu thêm Baseline đã làm sạch
        self.clean_baseline_ = BaselineWeekdayModel().fit(self.train_c_)

        fit_df = enrich_data(
            self.train_c_,
            self.stores_c_,
            self.weather_c_,
            self.nat_hol_,
            self.evt_hol_,
            self.tet_dates_,
        )
        X_shared = build_shared_matrix(fit_df)
        self.shared_columns_ = list(X_shared.columns)

        X_store = pd.get_dummies(fit_df["store_id"], prefix="sid", dtype=float)
        self.store_columns_ = list(X_store.columns)
        X_all = pd.concat([X_store, X_shared], axis=1)

        w_fit = np.where(fit_df["has_exact_weather"].values, 1.0, 0.15)
        self.max_train_date_ = fit_df["date"].max()
        days_ago = (self.max_train_date_ - fit_df["date"]).dt.days.values

        self.shared_coefs_ = {}
        self.shared_intercepts_ = {}
        self.store_ints_ = {}
        self.u_T_ = {}

        preds_fit = []
        for target_mode in ["log1p", "log"]:
            y_fit = (
                np.log1p(self.train_c_["revenue"].values)
                if target_mode == "log1p"
                else np.log(self.train_c_["revenue"].values)
            )
            model = Ridge(alpha=self.ridge_alpha).fit(X_all, y_fit, sample_weight=w_fit)
            sh_coef = model.coef_[len(self.store_columns_) :]
            sh_int = float(model.intercept_)
            self.shared_coefs_[target_mode] = sh_coef
            self.shared_intercepts_[target_mode] = sh_int

            pred_sh_fit = X_shared.values @ sh_coef + sh_int
            res_fit = y_fit - pred_sh_fit

            fit_df["raw_res"] = res_fit
            cd_shock = fit_df.groupby(["city", "date"])["raw_res"].transform("mean")
            clean_res = np.where(fit_df["has_exact_weather"].values, res_fit, res_fit - 0.85 * cd_shock.values)

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
            seen_stores = counts[counts > 0].index.tolist()
            w_meta = np.minimum(counts.loc[seen_stores].values, 200.0) / 200.0

            meta_X1 = pd.get_dummies(self.stores_c_[["city", "store_type"]], dtype=float)
            meta_X1["log_seats"] = (
                np.log(self.stores_c_["seats"]) - np.log(self.stores_c_["seats"]).mean()
            ).values
            meta_X1.index = self.stores_c_["store_id"]
            m_meta1 = Ridge(alpha=0.01).fit(meta_X1.loc[seen_stores], int_long.loc[seen_stores], sample_weight=w_meta)

            meta_X2 = meta_X1.copy()
            meta_X2["log_area"] = (
                np.log(self.stores_c_["area_m2"]) - np.log(self.stores_c_["area_m2"]).mean()
            ).values
            meta_X2["has_parking"] = self.stores_c_["has_parking"].astype(float).values
            m_meta2 = Ridge(alpha=0.05).fit(meta_X2.loc[seen_stores], int_long.loc[seen_stores], sample_weight=w_meta)

            pred_meta_int = pd.Series(
                0.50 * m_meta1.predict(meta_X1) + 0.50 * m_meta2.predict(meta_X2),
                index=self.stores_c_["store_id"],
            )

            final_store_int = pd.Series(
                {
                    sid: (counts.get(sid, 0) / (counts.get(sid, 0) + self.k_meta)) * emp_store_int.get(sid, 0.0)
                    + (self.k_meta / (counts.get(sid, 0) + self.k_meta)) * pred_meta_int[sid]
                    if counts.get(sid, 0) > 0
                    else pred_meta_int[sid]
                    for sid in self.stores_c_["store_id"]
                }
            )
            self.store_ints_[target_mode] = final_store_int

            dev_res = clean_res - fit_df["store_id"].map(final_store_int).values
            w_ar = (self.ar_rho ** days_ago) * (days_ago <= 21).astype(float) * w_fit
            tmp_ar = pd.DataFrame({"store_id": fit_df["store_id"].values, "war_res": dev_res * w_ar, "war": w_ar})
            g_ar = tmp_ar.groupby("store_id").sum()
            self.u_T_[target_mode] = (g_ar["war_res"] / (g_ar["war"] + 0.8)).reindex(self.stores_c_["store_id"]).fillna(0.0)

            p_fit = pred_sh_fit + fit_df["store_id"].map(final_store_int).values
            if target_mode == "log":
                ep_fit = np.exp(p_fit)
                sigma2 = 0.105 ** 2
                p_fit = np.log1p(ep_fit) + 0.5 * sigma2 * ep_fit / ((1.0 + ep_fit) ** 2)
            preds_fit.append(p_fit)

        p_stage1_fit = self.w_log1p * preds_fit[0] + (1.0 - self.w_log1p) * preds_fit[1]

        y_log1p = np.log1p(self.train_c_["revenue"].values)
        X_gbdt_fit = build_gbdt_matrix(fit_df, p_stage1_fit, self.store_ints_["log1p"])
        self.gbdt_columns_ = list(X_gbdt_fit.columns)

        self.hgb_ = HistGradientBoostingRegressor(
            max_iter=80,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=60,
            l2_regularization=20.0,
            random_state=self.random_state,
        )
        self.hgb_.fit(X_gbdt_fit, y_log1p - p_stage1_fit, sample_weight=w_fit)
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        """Dự báo doanh thu (triệu VNĐ) cho bảng dữ liệu đầu vào (date, store_id, is_promo)."""
        val_df = enrich_data(
            target_df[["date", "store_id", "is_promo"]],
            self.stores_c_,
            self.weather_c_,
            self.nat_hol_,
            self.evt_hol_,
            self.tet_dates_,
        )
        X_shared_val = build_shared_matrix(val_df, shared_columns=self.shared_columns_)
        h_days = np.maximum((val_df["date"] - self.max_train_date_).dt.days.values, 1)

        preds_val = []
        for target_mode in ["log1p", "log"]:
            sh_coef = self.shared_coefs_[target_mode]
            sh_int = self.shared_intercepts_[target_mode]
            final_store_int = self.store_ints_[target_mode]
            u_T = self.u_T_[target_mode]

            pred_sh_val = X_shared_val.values @ sh_coef + sh_int
            ar_corr = self.ar_scale * (self.ar_rho ** h_days) * val_df["store_id"].map(u_T).values
            p_val = pred_sh_val + val_df["store_id"].map(final_store_int).values + ar_corr

            if target_mode == "log":
                ep_val = np.exp(p_val)
                sigma2 = 0.105 ** 2
                p_val = np.log1p(ep_val) + 0.5 * sigma2 * ep_val / ((1.0 + ep_val) ** 2)
            preds_val.append(p_val)

        p_stage1_val = self.w_log1p * preds_val[0] + (1.0 - self.w_log1p) * preds_val[1]
        X_gbdt_val = build_gbdt_matrix(
            val_df,
            p_stage1_val,
            self.store_ints_["log1p"],
            gbdt_columns=self.gbdt_columns_,
        )
        res_gbdt = self.hgb_.predict(X_gbdt_val)

        pred_revenue = np.expm1(p_stage1_val + self.gbdt_weight * res_gbdt)
        return np.clip(pred_revenue, 0.0, None)

    def save(self, filepath: str) -> str:
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "CafeRevenueModel":
        return joblib.load(filepath)


def build_test_grid_if_missing(
    stores_df: pd.DataFrame,
    sample_sub_df: pd.DataFrame,
) -> pd.DataFrame:
    """Dựng lưới test Q1/2026 (3.869 dòng) khớp chính xác sample_submission.csv."""
    st_tmp = stores_df.assign(st=stores_df["store_type"].str.lower().str.strip())
    active = st_tmp[st_tmp["store_id"] != "S02"][["store_id", "st"]]
    grid = (
        pd.MultiIndex.from_product(
            [pd.date_range("2026-01-01", "2026-03-31"), active["store_id"]],
            names=["date", "store_id"],
        )
        .to_frame()
        .reset_index(drop=True)
    )
    grid = grid.merge(active, on="store_id").sort_values(["date", "store_id"]).reset_index(drop=True)
    tet_off = (grid["date"] - pd.Timestamp("2026-02-17")).dt.days
    closed = (
        ((grid["st"] == "office") & tet_off.between(-2, 2))
        | ((grid["st"] == "university") & tet_off.between(-2, 3))
    )
    test_df = grid[~closed][["date", "store_id"]].reset_index(drop=True)
    test_df.insert(0, "id", sample_sub_df["id"].values)
    test_df["is_promo"] = False
    return test_df
