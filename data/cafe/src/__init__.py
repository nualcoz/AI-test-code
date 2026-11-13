"""
src package initialization
Xuất các lớp và hàm chính của hệ thống dự báo doanh thu Cafe:
- Config: cấu hình đường dẫn và mốc kiểm định
- BaselineWeekdayModel, weekday_median: mô hình baseline đối chứng
- CafeRevenueModelV3: mô hình tối ưu thế hệ 3
- clean_stores, clean_weather, clean_holidays, clean_train, clean_test, build_test_grid_if_missing: pipeline làm sạch
- enrich_data, build_shared_matrix: pipeline đặc trưng dùng trong notebook phân tích
"""

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    SUBMISSIONS_DIR,
    REPORTS_DIR,
    TRAIN_PATH,
    TEST_PATH,
    STORES_PATH,
    WEATHER_PATH,
    HOLIDAYS_PATH,
    SAMPLE_SUB_PATH,
    CAFE_MODEL_V3_PATH,
    BASELINE_MODEL_PATH,
    SUBMISSION_V3_PATH,
    SUBMISSION_BASELINE_PATH,
    EVAL_QUARTERS,
)
from src.cleaning import (
    clean_stores,
    clean_weather,
    clean_holidays,
    clean_train,
    clean_test,
    build_test_grid_if_missing,
)
from src.features import (
    enrich_data,
    build_shared_matrix,
)
from src.baseline_model import (
    BaselineWeekdayModel,
    weekday_median,
)
from src.cafe_model_v3 import CafeRevenueModelV3

__all__ = [
    "DATA_DIR",
    "MODELS_DIR",
    "SUBMISSIONS_DIR",
    "REPORTS_DIR",
    "TRAIN_PATH",
    "TEST_PATH",
    "STORES_PATH",
    "WEATHER_PATH",
    "HOLIDAYS_PATH",
    "SAMPLE_SUB_PATH",
    "CAFE_MODEL_V3_PATH",
    "BASELINE_MODEL_PATH",
    "SUBMISSION_V3_PATH",
    "SUBMISSION_BASELINE_PATH",
    "EVAL_QUARTERS",
    "clean_stores",
    "clean_weather",
    "clean_holidays",
    "clean_train",
    "clean_test",
    "build_test_grid_if_missing",
    "enrich_data",
    "build_shared_matrix",
    "BaselineWeekdayModel",
    "weekday_median",
    "CafeRevenueModelV3",
]
