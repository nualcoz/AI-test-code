# -*- coding: utf-8 -*-
"""
src/config.py
Cấu hình đường dẫn, siêu tham số và phân chia kiểm thử cho dự án Dự báo Doanh thu Cafe.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

# Đảm bảo terminal console hiển thị đúng tiếng Việt UTF-8 trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Thư mục gốc dự án
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SUBMISSIONS_DIR = os.path.join(PROJECT_ROOT, "submissions")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, "notebooks")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def find_data_file(filename: str) -> str:
    """Tự động tìm kiếm file dữ liệu trong data/, project root, hoặc thư mục làm việc hiện tại."""
    candidates = [
        os.path.join(DATA_DIR, filename),
        os.path.join(PROJECT_ROOT, filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "data", filename),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    # Thử tìm trên Kaggle
    kaggle_matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if kaggle_matches:
        return os.path.abspath(kaggle_matches[0])
    return os.path.join(DATA_DIR, filename)


# Đường dẫn các file dữ liệu
TRAIN_PATH = find_data_file("train.csv")
TEST_PATH = find_data_file("test.csv")
STORES_PATH = find_data_file("stores.csv")
WEATHER_PATH = find_data_file("weather.csv")
HOLIDAYS_PATH = find_data_file("holidays.csv")
SAMPLE_SUB_PATH = find_data_file("sample_submission.csv")

# Đường dẫn file mô hình và kết quả
CAFE_MODEL_PATH = os.path.join(MODELS_DIR, "cafe_revenue_model.pkl")
CAFE_MODEL_V2_PATH = os.path.join(MODELS_DIR, "cafe_revenue_model_v2.pkl")
CAFE_MODEL_V3_PATH = os.path.join(MODELS_DIR, "cafe_revenue_model_v3.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, "submission.csv")
SUBMISSION_V2_PATH = os.path.join(SUBMISSIONS_DIR, "submission_v2.csv")
SUBMISSION_V3_PATH = os.path.join(SUBMISSIONS_DIR, "submission_v3.csv")
SUBMISSION_BASELINE_PATH = os.path.join(SUBMISSIONS_DIR, "submission_baseline.csv")

# Siêu tham số mặc định của CafeRevenueModel
MODEL_PARAMS = {
    "ridge_alpha": 0.5,
    "hl_long": 365.0,
    "hl_mid": 50.0,
    "w_long": 0.25,
    "k_meta": 2.0,
    "ar_rho": 0.80,
    "ar_scale": 0.65,
    "w_log1p": 0.25,
    "gbdt_weight": 0.10,
    "random_state": 42,
}

# Các mốc phân chia kiểm định chéo theo quý năm 2025
EVAL_QUARTERS = {
    "Q1/2025": ("2025-01-01", "2025-03-31"),
    "Q2/2025": ("2025-04-01", "2025-06-30"),
    "Q3/2025": ("2025-07-01", "2025-09-30"),
    "Q4/2025": ("2025-10-01", "2025-12-31"),
}
