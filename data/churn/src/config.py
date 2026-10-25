# -*- coding: utf-8 -*-
"""
src/config.py
Cấu hình đường dẫn, siêu tham số và phân chia kiểm thử cho dự án Card Churn Prediction.
"""

import os
import sys
import glob

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
    kaggle_matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if kaggle_matches:
        return os.path.abspath(kaggle_matches[0])
    return os.path.join(DATA_DIR, filename)


# Đường dẫn file dữ liệu
TRAIN_PATH = find_data_file("train.csv")
SAMPLE_SUB_PATH = find_data_file("sample_submission.csv")
TEST_PATH = find_data_file("test.csv")

# Đường dẫn file mô hình và kết quả
CHURN_MODEL_PATH = os.path.join(MODELS_DIR, "churn_model.pkl")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")
SUBMISSION_PATH = os.path.join(SUBMISSIONS_DIR, "submission.csv")
SUBMISSION_BASELINE_PATH = os.path.join(SUBMISSIONS_DIR, "submission_baseline.csv")

# Siêu tham số mặc định của ChurnPredictor
MODEL_PARAMS = {
    "n_seeds": 5,
    "gam_weight": 0.50,
    "random_state": 42,
}

# Cấu hình kiểm định chéo (Cross Validation)
CV_CONFIG = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 0,
}
