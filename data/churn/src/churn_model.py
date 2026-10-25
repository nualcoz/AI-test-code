# -*- coding: utf-8 -*-
"""
src/churn_model.py
Mô hình AI dự báo xác suất rời bỏ thẻ tín dụng (Card Churn):
- Tổ hợp Hybrid Ensemble 2 nhánh mô hình học máy bổ trợ nhau:
  1. Multi-seed HistGradientBoostingClassifier (5 seeds): GBDT xử lý tự nhiên giá trị thiếu,
     tương tác phi tuyến cục bộ bậc thấp (max_leaf_nodes=6).
  2. SplineTransformer GAM LogisticRegression: Mô hình cộng tính tổng quát với B-spline bậc 3
     bắt trọn đường cong phi tuyến trơn của age, utilization, spend_change_pct, tenure_days
     và các tương tác nghiệp vụ (kỷ niệm mở thẻ, phí thường niên, săn hoàn tiền).
- 5-Fold StratifiedKFold CV ROC-AUC: ~ 0.8129 (Tăng +8.39% so với Baseline 0.7290).
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from src.cleaning import ChurnDataCleaner
from src.features import (
    CAT_COLS,
    GAM_BIN_COLS,
    HGB_NUM_COLS,
    SPLINE_COLS,
    build_features,
)


class ChurnPredictor(BaseEstimator, ClassifierMixin):
    """Mô hình dự báo xác suất đóng thẻ (Card Churn) kết hợp GBDT đa seed & Spline GAM."""

    def __init__(
        self,
        n_seeds: int = 5,
        gam_weight: float = 0.50,
        random_state: int = 42,
    ):
        self.n_seeds = n_seeds
        self.gam_weight = gam_weight
        self.random_state = random_state
        self.hgb_models_ = []
        self.gam_model_ = None
        self.is_fitted = False

    def _make_hgb(self, seed: int) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_iter=450,
            learning_rate=0.025,
            max_leaf_nodes=6,
            min_samples_leaf=30,
            l2_regularization=10.0,
            random_state=seed,
        )

    def _make_gam(self):
        pre = ColumnTransformer(
            transformers=[
                (
                    "sp",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        SplineTransformer(n_knots=8, degree=3),
                        StandardScaler(),
                    ),
                    SPLINE_COLS,
                ),
                (
                    "bin",
                    make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                    ),
                    GAM_BIN_COLS,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                    CAT_COLS,
                ),
            ]
        )
        return make_pipeline(pre, LogisticRegression(C=0.12, max_iter=2000))

    def fit(self, clean_df: pd.DataFrame, y: pd.Series = None):
        """Huấn luyện tổ hợp HGB đa seed + Spline GAM trên dữ liệu sạch."""
        feat_df = build_features(clean_df)
        target = y if y is not None else feat_df["churn_3m"]

        hgb_features = HGB_NUM_COLS + CAT_COLS
        self.hgb_models_ = []
        for i in range(self.n_seeds):
            m = self._make_hgb(self.random_state + i)
            m.fit(feat_df[hgb_features], target)
            self.hgb_models_.append(m)

        self.gam_model_ = self._make_gam()
        self.gam_model_.fit(feat_df, target)
        self.is_fitted = True
        return self

    def predict_proba_components(self, clean_df: pd.DataFrame):
        """Trả về xác suất dự báo của từng nhánh (p_hgb, p_gam, p_ensemble)."""
        if not self.is_fitted:
            raise RuntimeError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        feat_df = build_features(clean_df)
        hgb_features = HGB_NUM_COLS + CAT_COLS

        p_hgb = np.zeros(len(feat_df))
        for m in self.hgb_models_:
            p_hgb += m.predict_proba(feat_df[hgb_features])[:, 1] / len(self.hgb_models_)

        p_gam = self.gam_model_.predict_proba(feat_df)[:, 1]
        p_ens = (1.0 - self.gam_weight) * p_hgb + self.gam_weight * p_gam
        return p_hgb, p_gam, p_ens

    def predict_proba(self, clean_df: pd.DataFrame) -> np.ndarray:
        """Dự báo xác suất rời bỏ [P(churn=0), P(churn=1)]."""
        _, _, p_ens = self.predict_proba_components(clean_df)
        p_ens = np.clip(p_ens, 0.0, 1.0)
        return np.column_stack([1.0 - p_ens, p_ens])

    def predict(self, clean_df: pd.DataFrame) -> np.ndarray:
        """Trả về xác suất rời bỏ P(churn=1)."""
        return self.predict_proba(clean_df)[:, 1]


class ChurnPipeline:
    """End-to-End Pipeline hoàn chỉnh cho bài toán Card Churn Prediction."""

    def __init__(
        self,
        n_seeds: int = 5,
        gam_weight: float = 0.50,
        random_state: int = 42,
    ):
        self.cleaner = ChurnDataCleaner()
        self.predictor = ChurnPredictor(
            n_seeds=n_seeds, gam_weight=gam_weight, random_state=random_state
        )
        self.is_fitted = False

    def fit(self, train_raw: pd.DataFrame):
        """Làm sạch dữ liệu huấn luyện thô và huấn luyện mô hình dự báo."""
        clean_df, _ = self.cleaner.clean(train_raw, deduplicate=True)
        self.predictor.fit(clean_df, clean_df["churn_3m"])
        self.is_fitted = True
        return self

    def clean(self, df_raw: pd.DataFrame, deduplicate: bool = True):
        """Làm sạch dữ liệu và trả về (clean_df, errors_found_df)."""
        if not self.is_fitted:
            self.cleaner.fit(df_raw)
        return self.cleaner.clean(df_raw, deduplicate=deduplicate)

    def predict(self, test_raw: pd.DataFrame) -> np.ndarray:
        """Dự báo xác suất churn_3m cho tập dữ liệu khách hàng mới."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline chưa được huấn luyện. Hãy gọi fit() trước.")
        test_clean = self.cleaner.transform(test_raw, deduplicate=False)
        return np.round(self.predictor.predict(test_clean), 6)

    def save(self, filepath: str):
        """Lưu toàn bộ pipeline ra file .pkl."""
        joblib.dump(self, filepath)
        return filepath

    @classmethod
    def load(cls, filepath: str) -> "ChurnPipeline":
        """Nạp pipeline từ file .pkl."""
        return joblib.load(filepath)
