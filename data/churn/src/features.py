# -*- coding: utf-8 -*-
"""
src/features.py
Feature Engineering & Biến đổi tương tác nghiệp vụ ngân hàng cho bài toán Card Churn:
1. Hiệu ứng Kỷ niệm ngày mở thẻ rơi vào 3 tháng tới (Tháng 7-9) + Không được miễn phí thường niên (anniversary_unwaived).
2. Hiệu ứng Khách săn khuyến mãi thẻ hoàn tiền vừa kết thúc ưu đãi 60 ngày (is_cashback_promo).
3. Chỉ báo hành vi: Không phát sinh số dư (util_is_zero), mức độ sụt giảm chi tiêu (spend_drop), thẻ mới mở (is_new_card).
4. Các tỷ số tài chính: spend_to_limit, spend_to_income, limit_to_income, avg_txn_amt.
5. Biến đổi Logarit trơn cho các biến lệch đuôi dài (income, limit, txn_amount, txn_count).
"""

import numpy as np
import pandas as pd

CAT_COLS = [
    "gender",
    "occupation",
    "card_type",
    "salary_via_bank",
    "last_contact_channel",
]

HGB_NUM_COLS = [
    "age",
    "age_imputed",
    "income_month",
    "income_imputed",
    "credit_limit",
    "annual_fee_waived",
    "utilization_6m",
    "util_is_zero",
    "txn_count_3m",
    "txn_amount_3m",
    "txn_amt_imputed",
    "avg_txn_amt",
    "spend_change_pct",
    "app_logins_30d",
    "app_logins_is_na",
    "complaints_12m",
    "late_payments_12m",
    "other_bank_cards",
    "promo_ended_60d",
    "tenure_days",
    "open_month",
    "is_anniversary_3m",
    "anniversary_unwaived",
    "is_cashback_promo",
]

SPLINE_COLS = [
    "age_imputed",
    "log_income",
    "log_limit",
    "utilization_6m",
    "log_txn_cnt",
    "log_txn_amt",
    "spend_change_pct",
    "app_logins_30d",
    "tenure_days",
]

GAM_BIN_COLS = [
    "annual_fee_waived",
    "complaints_12m",
    "late_payments_12m",
    "other_bank_cards",
    "promo_ended_60d",
    "is_anniversary_3m",
    "anniversary_unwaived",
    "is_cashback_promo",
    "util_is_zero",
    "spend_drop",
    "is_new_card",
    "app_logins_is_na",
    "other_cards_is_na",
]


def build_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất các đặc trưng phi tuyến và tương tác nghiệp vụ ngân hàng."""
    d = clean_df.copy()

    # 1. Kỷ niệm ngày mở thẻ rơi vào 3 tháng tới (Tháng 7, 8, 9) & Phí thường niên
    d["is_anniversary_3m"] = d["open_month"].isin([7, 8, 9]).astype(int)
    d["anniversary_unwaived"] = (
        (d["is_anniversary_3m"] == 1) & (d["annual_fee_waived"] == 0)
    ).astype(int)

    # 2. Khách săn khuyến mãi thẻ Cashback vừa hết ưu đãi 60 ngày
    d["is_cashback_promo"] = (
        (d["card_type"] == "Cashback") & (d["promo_ended_60d"] == 1)
    ).astype(int)

    # 3. Đặc trưng hành vi sử dụng thẻ & sụt giảm chi tiêu
    d["util_is_zero"] = (d["utilization_6m"] == 0).astype(int)
    d["spend_drop"] = np.minimum(d["spend_change_pct"].fillna(0.0), 0.0)
    d["is_new_card"] = (d["tenure_days"] <= 95).astype(int)

    # 4. Tỷ số tài chính & biến đổi Log
    d["avg_txn_amt"] = d["txn_amt_imputed"] / d["txn_count_3m"].clip(lower=1)
    d["spend_to_limit"] = d["txn_amt_imputed"] / (3.0 * d["credit_limit"].clip(lower=1))
    d["spend_to_income"] = d["txn_amt_imputed"] / (3.0 * d["income_imputed"].clip(lower=1))
    d["limit_to_income"] = d["credit_limit"] / d["income_imputed"].clip(lower=1)

    d["log_income"] = np.log1p(d["income_imputed"].clip(lower=0))
    d["log_limit"] = np.log1p(d["credit_limit"].clip(lower=0))
    d["log_txn_amt"] = np.log1p(d["txn_amt_imputed"].clip(lower=0))
    d["log_txn_cnt"] = np.log1p(d["txn_count_3m"].clip(lower=0))

    for c in CAT_COLS:
        d[c] = d[c].astype(str).astype("category")

    return d
