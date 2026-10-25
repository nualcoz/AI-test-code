"""
inspect_model.py
Script xem va kham pha noi dung ben trong cac file models (.pkl).

Cach dung:
    python inspect_model.py
"""

import sys
import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import CAFE_MODEL_PATH, BASELINE_MODEL_PATH


def inspect_baseline():
    print("=" * 80)
    print("1. NOI DUNG FILE: baseline_model.pkl (Mo Hinh Baseline Trung Vi)")
    print("=" * 80)
    base = joblib.load(BASELINE_MODEL_PATH)
    print(f"Kieu du lieu: {type(base)}")
    print(f"Doanh thu trung vi toan he thong (global_med_): {base.global_med_:.2f} trieu VNĐ")
    print(f"Tong so quan da hoc: {len(base.store_med_)} quan")
    
    # In mau 10 gia tri trung vi theo (quan, thu)
    print("\nVi du 8 cap (Quan, Thu trong tuan) -> Doanh thu trung vi:")
    sample_keys = list(base.store_dow_med_.keys())[:8]
    dow_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    for sid, dow in sample_keys:
        rev = base.store_dow_med_[(sid, dow)]
        print(f"  + Quan {sid} vao {dow_names[dow]}: {rev:.2f} trieu VNĐ")


def inspect_cafe_model():
    print("\n" + "=" * 80)
    print("2. NOI DUNG FILE: cafe_revenue_model.pkl (Mo Hinh AI Toi Uu)")
    print("=" * 80)
    model = joblib.load(CAFE_MODEL_PATH)
    print(f"Kieu du lieu: {type(model)}")
    print("\n[A] Thong ke cac loi du lieu mo hinh da phat hien & sua chua:")
    for k, v in model.clean_stats_.items():
        print(f"  + {k}: {v}")

    print("\n[B] He so co so (Base Intercept) cua top 5 quan doanh thu cao nhat & thap nhat:")
    store_ints = model.store_ints_["log1p"]
    top_stores = store_ints.sort_values(ascending=False)
    print("  -- Top 3 quan doanh thu cao nhat:")
    for sid, val in top_stores.head(3).items():
        print(f"     * {sid}: exp({val:.3f}) ~ {np.expm1(val):.2f} trieu/ngay")
    print("  -- Top 3 quan doanh thu khiem ton nhat:")
    for sid, val in top_stores.tail(3).items():
        print(f"     * {sid}: exp({val:.3f}) ~ {np.expm1(val):.2f} trieu/ngay")

    print("\n[C] He so tac dong chung (Shared Coefficients - Ridge Regression):")
    coefs = pd.Series(model.shared_coefs_["log1p"], index=model.shared_columns_)
    
    # In tac dong khuyen mai
    promo_cols = [c for c in coefs.index if "promo" in c]
    print("  + Tac dong cua Khuyen mai (Promo) theo loai quan:")
    for c in promo_cols:
        print(f"     * {c}: {coefs[c]:+.4f} (tang khoang {np.expm1(coefs[c])*100:+.1f}%)")

    # In tac dong mua gio
    rain_cols = [c for c in coefs.index if "rain" in c]
    print("  + Tac dong cua Mua (Rain) theo loai quan:")
    for c in rain_cols:
        print(f"     * {c}: {coefs[c]:+.4f}")

    print(f"\n[D] Mo hinh tang cuong HistGradientBoosting (Stage 2):")
    print(f"  + So dac trung dau vao: {len(model.gbdt_columns_)}")
    print(f"  + Trong so dieu chinh thoi gian AR(1) rho: {model.ar_rho}")
    print(f"  + Trong so GBDT weight: {model.gbdt_weight}")


def main():
    inspect_baseline()
    inspect_cafe_model()
    print("\n" + "=" * 80)
    print(">>> KET THUC XEM NOI DUNG MODEL")
    print("=" * 80)


if __name__ == "__main__":
    main()
