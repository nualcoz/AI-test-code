# -*- coding: utf-8 -*-
"""
src/cleaning.py
Bộ làm sạch dữ liệu & Chuẩn hóa đặc trưng chuyên sâu cho bài toán Card Churn:
1. Loại bỏ dòng trùng lặp (120 dòng bị nhân bản trong train.csv).
2. Chuẩn hóa các cột văn bản (gender, province, occupation, currency).
3. Chuẩn hóa ngày mở thẻ open_date (hỗ trợ cả 3 định dạng: YYYY-MM-DD, DD/MM/YYYY, YYYYMMDD).
4. Phát hiện và sửa lỗi đơn vị:
   - income_month >= 100,000 (VND thay vì triệu VND) -> chia 1e6
   - credit_limit >= 1,000,000 (VND thay vì triệu VND) -> chia 1e6
5. Lọc bỏ giá trị sentinel:
   - age < 18 hoặc age > 90 -> NaN
   - utilization_6m < 0 hoặc > 2.0 -> NaN
   - txn_amount_3m < 0 -> NaN
6. Điền khuyết chéo theo nghiệp vụ ngân hàng:
   - age <-> occupation
   - income_month <-> credit_limit theo card_type
   - txn_amount_3m theo tỷ lệ chi tiêu trung bình
   - Cờ báo khuyết thiếu cho app_logins_30d, other_bank_cards mang tín hiệu rời bỏ.
"""

import re
import numpy as np
import pandas as pd


class ChurnDataCleaner:
    """Bộ làm sạch dữ liệu & Chuẩn hóa đặc trưng chuyên sâu cho bài toán Card Churn."""

    SNAPSHOT_DATE = pd.Timestamp("2026-06-30")

    def __init__(self):
        self.occ_age_medians_ = {
            "student": 21.0,
            "retired": 66.0,
            "civil_servant": 37.0,
            "salaried": 37.0,
            "self_employed": 37.0,
            "other": 37.0,
            "MISSING": 38.0,
        }
        self.card_limit_income_ratios_ = {
            "Classic": 2.02,
            "Cashback": 2.51,
            "Gold": 3.03,
            "Platinum": 3.96,
        }
        self.avg_txn_income_ratio_ = 0.060
        self.num_medians_ = {}
        self.is_fitted = False

    @staticmethod
    def clean_gender(s: pd.Series) -> pd.Series:
        mapping = {
            "f": "F",
            "female": "F",
            "nữ": "F",
            "m": "M",
            "male": "M",
            "nam": "M",
        }
        cleaned = s.astype(str).str.strip().str.lower().map(mapping)
        return cleaned.fillna("MISSING")

    @staticmethod
    def clean_province(s: pd.Series) -> pd.Series:
        def _map_one(val):
            if pd.isna(val):
                return "MISSING"
            v = str(val).strip()
            vl = v.lower()
            if vl in ["tp.hcm", "tp hcm", "ho chi minh", "hồ chí minh", "hcm"]:
                return "TP.HCM"
            if vl in ["hà nội", "ha noi", "hn", "hanoi"]:
                return "Hà Nội"
            if vl in ["đà nẵng", "danang", "da nang"]:
                return "Đà Nẵng"
            if vl in ["thừa thiên huế", "hue", "huế"]:
                return "Thừa Thiên Huế"
            if vl in ["bà rịa - vũng tàu", "vũng tàu", "br-vt"]:
                return "Bà Rịa - Vũng Tàu"
            return v

        return s.map(_map_one)

    @staticmethod
    def clean_occupation(s: pd.Series) -> pd.Series:
        occ = (
            s.astype(str)
            .str.strip()
            .str.lower()
            .str.replace("-", "_", regex=False)
            .str.replace(" ", "_", regex=False)
        )
        return occ.replace({"nan": np.nan, "none": np.nan, "": np.nan})

    @staticmethod
    def parse_open_date(s: pd.Series) -> pd.Series:
        s_str = s.astype(str).str.strip()
        dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
        m1 = s_str.str.match(r"^\d{4}-\d{2}-\d{2}$")
        m2 = s_str.str.match(r"^\d{2}/\d{2}/\d{4}$")
        m3 = s_str.str.match(r"^\d{8}$")
        dt.loc[m1] = pd.to_datetime(s_str.loc[m1], format="%Y-%m-%d", errors="coerce")
        dt.loc[m2] = pd.to_datetime(s_str.loc[m2], format="%d/%m/%Y", errors="coerce")
        dt.loc[m3] = pd.to_datetime(s_str.loc[m3], format="%Y%m%d", errors="coerce")
        return dt

    def fit(self, df: pd.DataFrame):
        """Học các tham số thống kê để điền khuyết từ tập huấn luyện."""
        d = df.drop_duplicates(subset=["customer_id"], keep="first").copy()

        occ = self.clean_occupation(d["occupation"])
        age = d["age"].where((d["age"] >= 18) & (d["age"] <= 90), np.nan)
        occ = occ.where(~(occ.isna() & (age >= 65)), "retired")
        occ = occ.where(~(occ.isna() & (age <= 21)), "student")
        occ_filled = occ.fillna("MISSING")

        # 1. Trung vị tuổi theo nghề nghiệp
        age_meds = age.groupby(occ_filled).median().to_dict()
        for k, v in age_meds.items():
            if pd.notna(v):
                self.occ_age_medians_[k] = float(v)
        self.occ_age_medians_["MISSING"] = float(age.median())

        # 2. Tỷ lệ credit_limit / income_month theo card_type
        inc = d["income_month"].copy()
        inc.loc[inc >= 100_000] /= 1e6
        lim = d["credit_limit"].copy()
        lim.loc[lim >= 1_000_000] /= 1e6
        ratio = (lim / inc).groupby(d["card_type"]).median().to_dict()
        for k, v in ratio.items():
            if pd.notna(v):
                self.card_limit_income_ratios_[k] = float(v)

        # 3. Tỷ lệ (txn_amount_3m / txn_count_3m) / income_month
        txn_amt = d["txn_amount_3m"].where(d["txn_amount_3m"] >= 0, np.nan)
        avg_txn = txn_amt / d["txn_count_3m"].clip(lower=1)
        valid_ratio = (avg_txn / inc).dropna()
        if len(valid_ratio) > 0:
            self.avg_txn_income_ratio_ = float(valid_ratio.median())

        # 4. Lưu trung vị dự phòng cho các biến số
        dt = self.parse_open_date(d["open_date"])
        tenure_days = (self.SNAPSHOT_DATE - dt).dt.days
        util = d["utilization_6m"].where(
            (d["utilization_6m"] >= 0) & (d["utilization_6m"] <= 2.0), np.nan
        )
        self.num_medians_ = {
            "age": float(age.median()),
            "income_month": float(inc.median()),
            "credit_limit": float(lim.median()),
            "utilization_6m": float(util.median()),
            "txn_amount_3m": float(txn_amt.median()),
            "app_logins_30d": float(d["app_logins_30d"].median()),
            "other_bank_cards": float(d["other_bank_cards"].median()),
            "tenure_days": float(tenure_days.median()),
        }
        self.is_fitted = True
        return self

    def find_errors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lập bảng thống kê chi tiết tất cả các dòng bị lỗi trong dữ liệu gốc."""
        records = []

        # 1. Trùng lặp
        dup_mask = df.duplicated(subset=["customer_id"], keep="first")
        for idx, row in df[dup_mask].iterrows():
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "customer_id",
                "error_type": "duplicate_row",
                "raw_value": str(row["customer_id"]),
                "cleaned_value": "DROPPED",
            })

        d = df[~dup_mask].copy()

        # 2. Lỗi tuổi (age < 18 hoặc age > 90)
        bad_age = d["age"].notna() & ((d["age"] < 18) | (d["age"] > 90))
        for idx, row in d[bad_age].iterrows():
            occ_c = self.clean_occupation(pd.Series([row["occupation"]])).iloc[0]
            occ_key = occ_c if pd.notna(occ_c) else "MISSING"
            new_age = self.occ_age_medians_.get(occ_key, 38.0)
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "age",
                "error_type": "invalid_sentinel_age",
                "raw_value": str(row["age"]),
                "cleaned_value": str(round(new_age, 1)),
            })

        # 3. Lỗi đơn vị thu nhập (income_month >= 100,000)
        bad_inc = d["income_month"].notna() & (d["income_month"] >= 100_000)
        for idx, row in d[bad_inc].iterrows():
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "income_month",
                "error_type": "unit_error_vnd_x1e6",
                "raw_value": str(row["income_month"]),
                "cleaned_value": str(round(row["income_month"] / 1e6, 2)),
            })

        # 4. Lỗi đơn vị hạn mức (credit_limit >= 1,000,000)
        bad_lim = d["credit_limit"].notna() & (d["credit_limit"] >= 1_000_000)
        for idx, row in d[bad_lim].iterrows():
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "credit_limit",
                "error_type": "unit_error_vnd_x1e6",
                "raw_value": str(row["credit_limit"]),
                "cleaned_value": str(round(row["credit_limit"] / 1e6, 2)),
            })

        # 5. Lỗi utilization_6m (< 0 hoặc > 2.0)
        bad_util = d["utilization_6m"].notna() & (
            (d["utilization_6m"] < 0) | (d["utilization_6m"] > 2.0)
        )
        for idx, row in d[bad_util].iterrows():
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "utilization_6m",
                "error_type": "invalid_sentinel_utilization",
                "raw_value": str(row["utilization_6m"]),
                "cleaned_value": "NaN -> Imputed",
            })

        # 6. Lỗi txn_amount_3m âm (< 0)
        bad_txn = d["txn_amount_3m"].notna() & (d["txn_amount_3m"] < 0)
        for idx, row in d[bad_txn].iterrows():
            records.append({
                "row_index": idx,
                "customer_id": row["customer_id"],
                "column": "txn_amount_3m",
                "error_type": "negative_txn_amount",
                "raw_value": str(row["txn_amount_3m"]),
                "cleaned_value": "NaN -> Imputed",
            })

        return pd.DataFrame(records)

    def transform(self, df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
        """Làm sạch dữ liệu và tạo các cột đã xử lý."""
        if not self.is_fitted:
            self.fit(df)

        d = df.copy()
        if deduplicate and "customer_id" in d.columns:
            d = d.drop_duplicates(subset=["customer_id"], keep="first").reset_index(drop=True)

        # 1. Chuẩn hóa văn bản
        d["gender"] = self.clean_gender(d["gender"])
        d["province"] = self.clean_province(d["province"])
        occ = self.clean_occupation(d["occupation"])
        d["currency"] = d["currency"].astype(str).str.strip().str.upper()

        # 2. Xử lý tuổi và suy luận chéo với nghề nghiệp
        d["age_is_na"] = d["age"].isna().astype(int)
        d["age_is_err"] = ((d["age"] < 18) | (d["age"] > 90)).astype(int)
        d.loc[(d["age"] < 18) | (d["age"] > 90), "age"] = np.nan

        occ = occ.where(~(occ.isna() & (d["age"] >= 65)), "retired")
        occ = occ.where(~(occ.isna() & (d["age"] <= 21)), "student")
        d["occupation"] = occ.fillna("MISSING")

        d["age_imputed"] = d["age"].fillna(
            d["occupation"].map(self.occ_age_medians_)
        ).fillna(self.num_medians_["age"])

        # 3. Xử lý lỗi đơn vị thu nhập & hạn mức tín dụng
        d["income_unit_err"] = (d["income_month"] >= 100_000).astype(int)
        d.loc[d["income_month"] >= 100_000, "income_month"] /= 1e6

        d["limit_unit_err"] = (d["credit_limit"] >= 1_000_000).astype(int)
        d.loc[d["credit_limit"] >= 1_000_000, "credit_limit"] /= 1e6

        d["income_is_na"] = d["income_month"].isna().astype(int)
        est_income = d["credit_limit"] / d["card_type"].map(self.card_limit_income_ratios_)
        d["income_imputed"] = (
            d["income_month"]
            .fillna(est_income)
            .fillna(self.num_medians_["income_month"])
            .round(2)
        )

        # 4. Xử lý utilization_6m và txn_amount_3m
        d["util_is_err"] = (
            (d["utilization_6m"] < 0) | (d["utilization_6m"] > 2.0)
        ).astype(int)
        d.loc[(d["utilization_6m"] < 0) | (d["utilization_6m"] > 2.0), "utilization_6m"] = np.nan
        d["utilization_imputed"] = d["utilization_6m"].fillna(self.num_medians_["utilization_6m"])

        d["txn_amt_is_neg"] = (d["txn_amount_3m"] < 0).astype(int)
        d.loc[d["txn_amount_3m"] < 0, "txn_amount_3m"] = np.nan
        est_txn_amt = (
            self.avg_txn_income_ratio_ * d["income_imputed"] * d["txn_count_3m"]
        ).round(2)
        d["txn_amt_imputed"] = (
            d["txn_amount_3m"]
            .fillna(est_txn_amt)
            .fillna(self.num_medians_["txn_amount_3m"])
        )

        # 5. Xử lý các cột có giá trị khuyết thiếu mang ý nghĩa hành vi
        d["app_logins_is_na"] = d["app_logins_30d"].isna().astype(int)
        d["app_logins_imputed"] = d["app_logins_30d"].fillna(self.num_medians_["app_logins_30d"])

        d["other_cards_is_na"] = d["other_bank_cards"].isna().astype(int)
        d["other_cards_imputed"] = d["other_bank_cards"].fillna(self.num_medians_["other_bank_cards"])

        d["salary_via_bank"] = d["salary_via_bank"].fillna("MISSING")
        d["last_contact_channel"] = d["last_contact_channel"].fillna("MISSING")

        # 6. Chuẩn hóa ngày mở thẻ & Tính thâm niên (Tenure)
        open_dt = self.parse_open_date(d["open_date"])
        d["open_date_clean"] = open_dt.dt.strftime("%Y-%m-%d")
        d["tenure_days"] = (self.SNAPSHOT_DATE - open_dt).dt.days
        d["tenure_months"] = (d["tenure_days"] / 30.4375).round(2)
        d["open_month"] = open_dt.dt.month
        d["tenure_days_imputed"] = d["tenure_days"].fillna(self.num_medians_["tenure_days"])

        return d

    def clean(self, df: pd.DataFrame, deduplicate: bool = True):
        """Thực hiện làm sạch và trả về (clean_df, errors_found_df)."""
        if not self.is_fitted:
            self.fit(df)
        errors_df = self.find_errors(df)
        clean_df = self.transform(df, deduplicate=deduplicate)
        return clean_df, errors_df
