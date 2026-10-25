# -*- coding: utf-8 -*-
"""
src package initialization
Xuất các lớp và hàm chính của hệ thống dự báo Card Churn:
- Config: cấu hình đường dẫn và tham số mô hình
- ChurnDataCleaner: pipeline làm sạch dữ liệu bẩn và điền khuyết nghiệp vụ
- build_features: pipeline trích xuất đặc trưng và tương tác phi tuyến
- BaselineLogisticModel: mô hình baseline chuẩn theo starter_churn.ipynb
- ChurnPredictor, ChurnPipeline: mô hình AI tối ưu leaderboard
"""

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    REPORTS_DIR,
    NOTEBOOKS_DIR,
    TRAIN_PATH,
    SAMPLE_SUB_PATH,
    TEST_PATH,
    CHURN_MODEL_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_BASELINE_PATH,
    MODEL_PARAMS,
    CV_CONFIG,
)
from src.cleaning import ChurnDataCleaner
from src.features import (
    build_features,
    CAT_COLS,
    HGB_NUM_COLS,
    SPLINE_COLS,
    GAM_BIN_COLS,
)
from src.baseline_model import BaselineLogisticModel
from src.churn_model import ChurnPredictor, ChurnPipeline

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "SUBMISSIONS_DIR",
    "REPORTS_DIR",
    "NOTEBOOKS_DIR",
    "TRAIN_PATH",
    "SAMPLE_SUB_PATH",
    "TEST_PATH",
    "CHURN_MODEL_PATH",
    "BASELINE_MODEL_PATH",
    "SUBMISSION_PATH",
    "SUBMISSION_BASELINE_PATH",
    "MODEL_PARAMS",
    "CV_CONFIG",
    "ChurnDataCleaner",
    "build_features",
    "CAT_COLS",
    "HGB_NUM_COLS",
    "SPLINE_COLS",
    "GAM_BIN_COLS",
    "BaselineLogisticModel",
    "ChurnPredictor",
    "ChurnPipeline",
]
