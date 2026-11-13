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

from src.config import CAFE_MODEL_V3_PATH, BASELINE_MODEL_PATH


def inspect_baseline():
    print("=" * 80)
    print("1. NOI DUNG FILE: baseline_model.pkl (Mo Hinh Baseline Trung Vi)")
    print("=" * 80)
    base = joblib.load(BASELINE_MODEL_PATH)
    print(f"Kieu du lieu: {type(base)}")
    print(f"Doanh thu trung vi toan he thong (global_med_): {base.global_med_:.2f} trieu VNĐ")
    print(f"Tong so quan da hoc: {len(base.store_med_)} quan")

    print("\nVi du 8 cap (Quan, Thu trong tuan) -> Doanh thu trung vi:")
    sample_keys = list(base.store_dow_med_.keys())[:8]
    dow_names = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
    for sid, dow in sample_keys:
        rev = base.store_dow_med_[(sid, dow)]
        print(f"  + Quan {sid} vao {dow_names[dow]}: {rev:.2f} trieu VNĐ")


def inspect_cafe_model():
    print("\n" + "=" * 80)
    print("2. NOI DUNG FILE: cafe_revenue_model_v3.pkl (Mo Hinh AI Toi Uu The He 3)")
    print("=" * 80)
    model = joblib.load(CAFE_MODEL_V3_PATH)
    print(f"Kieu du lieu: {type(model)}")
    print(f"Moc du lieu huan luyen cuoi (max_train_date_): {model.max_train_date_.date()}")

    print("\n[A] Thong ke cac loi du lieu mo hinh da phat hien & sua chua:")
    for k, v in model.clean_stats_.items():
        print(f"  + {k}: {v}")

    print(f"\n[B] He so co so (Intercept = sh_int_ {model.sh_int_:.3f} + offset) top 3 quan cao nhat & thap nhat:")
    top_stores = model.store_ints_.sort_values(ascending=False)
    print("  -- Top 3 quan doanh thu cao nhat:")
    for sid, val in top_stores.head(3).items():
        base_rev = np.expm1(model.sh_int_ + val)
        print(f"     * {sid}: {model.sh_int_:.3f} {val:+.3f} => muc nen ~ {base_rev:.2f} trieu/ngay")
    print("  -- Top 3 quan doanh thu khiem ton nhat:")
    for sid, val in top_stores.tail(3).items():
        base_rev = np.expm1(model.sh_int_ + val)
        print(f"     * {sid}: {model.sh_int_:.3f} {val:+.3f} => muc nen ~ {base_rev:.2f} trieu/ngay")

    print("\n[C] He so tac dong chung (Shared Coefficients - Ridge Regression):")
    coefs = pd.Series(model.sh_coef_, index=model.shared_cols_)

    promo_cols = [c for c in coefs.index if "promo" in c]
    print("  + Tac dong cua Khuyen mai (Promo) theo loai quan:")
    for c in promo_cols:
        print(f"     * {c}: {coefs[c]:+.4f} (tang khoang {np.expm1(coefs[c])*100:+.1f}%)")

    rain_cols = [c for c in coefs.index if "rain" in c]
    print("  + Tac dong cua Mua (Rain):")
    for c in rain_cols[:6]:
        print(f"     * {c}: {coefs[c]:+.4f}")

    print("\n[D] Mo hinh tang cuong HistGradientBoosting & Kalman AR(1):")
    print(f"  + So dac trung Ridge shared : {len(model.shared_cols_)}")
    print(f"  + So dac trung GBDT         : {len(model.gbdt_cols_)}")
    print(f"  + He so AR(1) rho / scale   : {model.ar_rho} / {model.ar_scale}")
    print(f"  + Trong so GBDT weight      : {model.gbdt_weight}")
    print(f"  + Hieu chuan Taylor sigma   : {model.sigma_noise:.4f}")


def main():
    inspect_baseline()
    inspect_cafe_model()
    print("\n" + "=" * 80)
    print(">>> KET THUC XEM NOI DUNG MODEL")
    print("=" * 80)


if __name__ == "__main__":
    main()
