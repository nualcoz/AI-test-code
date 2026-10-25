"""
src package initialization
Xuất các lớp và hàm chính của hệ thống dự báo doanh thu Cafe:
- Config: cấu hình đường dẫn và siêu tham số
- BaselineWeekdayModel, weekday_median: mô hình chuẩn theo baseline
- CafeRevenueModel, build_test_grid_if_missing: mô hình nâng cấp tối ưu leaderboard
- clean_stores, clean_weather, clean_holidays, clean_train: pipeline làm sạch
- enrich_data, build_shared_matrix, build_gbdt_matrix: pipeline đặc trưng
"""

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    REPORTS_DIR,
    TRAIN_PATH,
    STORES_PATH,
    WEATHER_PATH,
    HOLIDAYS_PATH,
    SAMPLE_SUB_PATH,
    CAFE_MODEL_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_PATH,
    SUBMISSION_BASELINE_PATH,
    MODEL_PARAMS,
    EVAL_QUARTERS,
)
from src.cleaning import (
    clean_stores,
    clean_weather,
    clean_holidays,
    clean_train,
)
from src.features import (
    enrich_data,
    build_shared_matrix,
    build_gbdt_matrix,
)
from src.baseline_model import (
    BaselineWeekdayModel,
    weekday_median,
)
from src.cafe_model import (
    CafeRevenueModel,
    build_test_grid_if_missing,
)

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "SUBMISSIONS_DIR",
    "REPORTS_DIR",
    "TRAIN_PATH",
    "STORES_PATH",
    "WEATHER_PATH",
    "HOLIDAYS_PATH",
    "SAMPLE_SUB_PATH",
    "CAFE_MODEL_PATH",
    "BASELINE_MODEL_PATH",
    "SUBMISSION_PATH",
    "SUBMISSION_BASELINE_PATH",
    "MODEL_PARAMS",
    "EVAL_QUARTERS",
    "clean_stores",
    "clean_weather",
    "clean_holidays",
    "clean_train",
    "enrich_data",
    "build_shared_matrix",
    "build_gbdt_matrix",
    "BaselineWeekdayModel",
    "weekday_median",
    "CafeRevenueModel",
    "build_test_grid_if_missing",
]
