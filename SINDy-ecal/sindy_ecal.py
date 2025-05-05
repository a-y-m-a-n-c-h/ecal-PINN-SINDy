import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysindy as ps
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  # Replace MinMaxScaler
from pysindy.differentiation import SmoothedFiniteDifference

# ────────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ────────────────────────────────────────────────────────────────────────────────
def load_crystal_csv(csv_path: Path):
    """
    Return:
      - scaled calibration  (1‑D array)
      - scaled luminosity   (1‑D array)
      - relative time (seconds, 1‑D)
      - calibration scaler  (for inverse‑transform)
    """
    df = (
        pd.read_csv(csv_path)
          .sort_values("laser_datetime")
          .reset_index(drop=True)
    )

    # Parse to datetime and convert to seconds since first shot
    t = pd.to_datetime(df["laser_datetime"])
    rel_t = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)

    cali = df[["calibration"]].values.astype(np.float64)
    lumi = df[["int_deliv_inv_ub"]].values.astype(np.float64)

    cali_scaler = StandardScaler()
    lumi_scaler = StandardScaler()

    cali_s = cali_scaler.fit_transform(cali).flatten()
    lumi_s = lumi_scaler.fit_transform(lumi).flatten()


    return cali_s, lumi_s, rel_t, cali_scaler

def save_plot(true, pred, title, save_path, dpi=300):
    plt.figure(figsize=(12, 6))  # Wider figure
    
    # Convert steps to days (assuming 45-min steps)
    days = np.arange(len(true)) * (45/60) / 24  # 45min → days
    
    plt.plot(days, true, label="True Calibration", linewidth=1.5, alpha=0.8)
    plt.plot(days, pred, label="Model Prediction", linestyle="--", linewidth=1.2)
    
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Calibration Value (Scaled)", fontsize=12)
    plt.title(title, fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

def save_metrics(true, pred, csv_path):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    df = pd.DataFrame({"MAE": [mae], "MSE": [mse]})
    df.to_csv(csv_path, index=False)
    return mae, mse

# ────────────────────────────────────────────────────────────────────────────────
# SINDy helpers
# ────────────────────────────────────────────────────────────────────────────────

def plot_calibration_vs_time(calibration, time_s, title="Calibration vs Time", save_path=None):
    """
    Plot the calibration signal as a function of time.

    Parameters:
    - calibration (1D np.array): Scaled calibration values
    - time_s (1D np.array): Relative time in seconds
    - title (str): Plot title
    - save_path (str or Path): If given, save the plot to this file instead of showing it
    """
    plt.figure(figsize=(10, 4))
    plt.plot(time_s, calibration, label="Scaled Calibration", linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("Scaled Calibration")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def train_sindy_discrete(calib, lumi,
                         degree: int = 2,
                         thresh: float = 1e-6):
    """
    Learn c_{k+1} = f(c_k, ℓ_k) by treating both c and ℓ as state inputs.
    """
    X = np.vstack([calib, lumi]).T.astype(np.float64)

    lib = ps.PolynomialLibrary(degree=degree, include_interaction=True)
    opt = ps.STLSQ(threshold=thresh)

    model = ps.SINDy(
        feature_library=lib,
        optimizer=opt,
        feature_names=["c", "l"],  
        discrete_time=True
    )
    model.fit(X)

    print("Discovered discrete-time map:")
    print(model.equations(precision=6))
    return model




def rollout_discrete(model, calib, lumi, case="case1"):
    """
    Iterate c_{k+1} = f(c_k, ℓ_k) using the learned map.
      - case1 = teacher-forcing (use ground‐truth c_k)
      - case2 = free-roll     (use model’s own prediction)
    """
    preds = np.empty_like(calib, dtype=np.float64)
    preds[0] = calib[0]

    for k in range(len(calib) - 1):
        c_k = calib[k] if case == "case1" else preds[k]
        xk = np.array([[c_k, lumi[k]]], dtype=np.float64)
        preds[k+1] = model.predict(xk)[0, 0]

    return preds

# ────────────────────────────────────────────────────────────────────────────────
# Main routine
# ────────────────────────────────────────────────────────────────────────────────

def main(args):
    data_root = Path(args.data_root).expanduser()
    crystal_tag = f"{args.crystal_id:05d}"

    # ─── Training ────────────────────────────────────────────────────────────
    train_csv = data_root / f"df_skimmed_xtal_{crystal_tag}_{args.train_year}.csv"
    c_train, l_train, t_train ,cali_scaler = load_crystal_csv(train_csv)

    sindy_model = train_sindy_discrete(c_train, l_train, degree=args.poly_deg, thresh=args.threshold)
    #plot_calibration_vs_time(c_train,t_train)

    eqn_path = train_csv.parent / f"sindy_model_{crystal_tag}.txt"
    with open(eqn_path, "w") as f:
        f.write("\n".join(sindy_model.equations(precision=5)))

    print(f"[✓] Discovered equation saved → {eqn_path}")

    # ─── Evaluation ──────────────────────────────────────────────────────────
    for year in args.test_years:
        test_csv = data_root / f"df_skimmed_xtal_{crystal_tag}_{year}.csv"
        c_test, l_test, t_test, _ = load_crystal_csv(test_csv)
        out_dir = test_csv.parent
        true_unscaled = cali_scaler.inverse_transform(c_test.reshape(-1,1)).ravel()

        # Case 1: teacher‐forcing
        pred_c1 = rollout_discrete(sindy_model, c_test, l_test, case="case1")
        c1 = cali_scaler.inverse_transform(pred_c1.reshape(-1,1)).ravel()
        save_metrics(true_unscaled, c1, out_dir/f"case1_{year}.csv")
        save_plot(true_unscaled, c1,
                  f"Crystal {crystal_tag} — {year} (DT Teacher)",
                  out_dir/f"case1_{year}.png")

        # Case 2: free‐roll
        pred_c2 = rollout_discrete(sindy_model, c_test, l_test, case="case2")
        c2 = cali_scaler.inverse_transform(pred_c2.reshape(-1,1)).ravel()
        save_metrics(true_unscaled, c2, out_dir/f"case2_{year}.csv")
        save_plot(true_unscaled, c2,
                  f"Crystal {crystal_tag} — {year} (DT Free-roll)",
                  out_dir/f"case2_{year}.png")
    print("[✓] All done.")

# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train & evaluate a SINDy model on ECAL crystal data.")
    p.add_argument("--data_root", type=str, required=True,
                   help="Directory containing df_skimmed_xtal_* CSV files")
    p.add_argument("--crystal_id", type=int, default=54000)
    p.add_argument("--train_year", type=int, default=2016)
    p.add_argument("--test_years", type=int, nargs="+", default=[2017, 2018])
    p.add_argument("--poly_deg", type=int, default=2,
                   help="Polynomial degree for the feature library")
    p.add_argument("--threshold", type=float, default=0.05,
                   help="Sparsity threshold (λ) for STLSQ")

    main(p.parse_args())

