# -*- coding: utf-8 -*-
"""
src/baseline_model.py
Mô hình Baseline theo starter_churn.ipynb:
- Chỉ sử dụng các cột số thô (numeric columns only).
- Điền khuyết tật bằng trung vị (median) của tập huấn luyện.
- Chuẩn hóa thang đo bằng StandardScaler.
- Hồi quy LogisticRegression(max_iter=2000).
- 5-Fold StratifiedKFold CV ROC-AUC: ~ 0.7290.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaselineLogisticModel(BaseEstimator, ClassifierMixin):
    """Mô hình Baseline chuẩn theo starter_churn.ipynb (Logistic Regression trên cột số thô)."""

    def __init__(self, max_iter: int = 2000, random_state: int = 0):
        self.max_iter = max_iter
        self.random_state = random_state
        self.num_cols_ = []
        self.fill_medians_ = {}
        self.pipeline_ = None

    def fit(self, train_df: pd.DataFrame, y: pd.Series = None):
        """Huấn luyện mô hình Baseline trên dữ liệu thô."""
        df = train_df.copy()
        if y is None:
            y = df["churn_3m"]

        self.num_cols_ = [
            c for c in df.select_dtypes("number").columns if c != "churn_3m"
        ]
        self.fill_medians_ = df[self.num_cols_].median().to_dict()

        X = df[self.num_cols_].fillna(self.fill_medians_)
        self.pipeline_ = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=self.max_iter, random_state=self.random_state),
        )
        self.pipeline_.fit(X, y)
        return self

    def predict_proba(self, target_df: pd.DataFrame) -> np.ndarray:
        """Dự báo xác suất rời bỏ [P(churn=0), P(churn=1)]."""
        df = target_df.copy()
        # Đảm bảo có đủ các cột số
        for c in self.num_cols_:
            if c not in df.columns:
                df[c] = self.fill_medians_.get(c, 0.0)

        X = df[self.num_cols_].fillna(self.fill_medians_)
        return self.pipeline_.predict_proba(X)

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        """Trả về xác suất churn_3m P(churn=1)."""
        return self.predict_proba(target_df)[:, 1]
