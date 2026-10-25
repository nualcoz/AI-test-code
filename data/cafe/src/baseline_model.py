"""
src/baseline_model.py
Mô hình Baseline theo đề bài starter_cafe.ipynb:
- Dự báo doanh thu bằng Trung vị theo (store_id, day_of_week).
- Fallback cấp 1: Trung vị của riêng quán (store_id).
- Fallback cấp 2: Trung vị toàn cục của toàn hệ thống (global median).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class BaselineWeekdayModel(BaseEstimator, RegressorMixin):
    """Mô hình Baseline chuẩn theo phương pháp Day-of-Week Median trong starter_cafe.ipynb."""

    def __init__(self):
        self.store_dow_med_ = {}
        self.store_med_ = {}
        self.global_med_ = 0.0

    def fit(self, train_df: pd.DataFrame, y=None):
        df = train_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["dow"] = df["date"].dt.dayofweek
        self.global_med_ = float(df["revenue"].median())
        self.store_med_ = df.groupby("store_id")["revenue"].median().to_dict()
        self.store_dow_med_ = (
            df.groupby(["store_id", "dow"])["revenue"].median().to_dict()
        )
        return self

    def predict(self, target_df: pd.DataFrame) -> np.ndarray:
        target = target_df.copy()
        target["date"] = pd.to_datetime(target["date"])
        dow = target["date"].dt.dayofweek
        keys = list(zip(target["store_id"], dow))
        p = pd.Series(keys).map(self.store_dow_med_)
        store_med = target["store_id"].map(self.store_med_)
        p = p.fillna(store_med.reset_index(drop=True))
        return p.fillna(self.global_med_).values


def weekday_median(history: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Hàm tiện ích tương đương Baseline gốc từ starter_cafe.ipynb."""
    return BaselineWeekdayModel().fit(history).predict(target)
