"""
Follow-up checks for counterpress_wc2022.py
Run from the repository root:  python diagnostics/extra_checks.py
"""
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import counterpress_wc2022 as cp      # reuses the cached counterpress_dataset.csv

RUN_NO_VIS_CV = True     # check 2 (takes a few minutes: nested CV again)
RUN_REGAIN = False       # check 3 (full run for y_regain; opens a SHAP plot at the end)

df = cp.build_dataset()
df = df[df["nA_10"].notna()].reset_index(drop=True)
print(df.shape, "positive rate:", df["y_cp"].mean().round(3))

# ---- Check 1: how good is A_min_dist ALONE? ------------------------------------------
d = df["A_min_dist"].fillna(df["A_min_dist"].max())
print("\n[1] single feature -A_min_dist")
print("    ROC-AUC:", round(roc_auc_score(df["y_cp"], -d), 3),
      "| PR-AUC:", round(average_precision_score(df["y_cp"], -d), 3),
      "| base rate:", round(df["y_cp"].mean(), 3))

# ---- Check 2: drop camera-coverage-sensitive features and re-run the CV ---------------
if RUN_NO_VIS_CV:
    drop = ["match_id", "possession", "y_cp", "y_regain", "win_type", "gk_win"]
    X = pd.get_dummies(df.drop(columns=drop), columns=["loss_type", "prev_play_pattern"])
    vis_cols = [c for c in X.columns if c.startswith(("vis_", "hull_", "std_"))]
    print("\n[2] dropping", len(vis_cols), "coverage-sensitive features:", vis_cols)
    res = cp.nested_cv(X.drop(columns=vis_cols), df["y_cp"], df["match_id"])
    print(res.groupby("model")[["roc_auc", "pr_auc", "mcc", "bal_acc"]]
             .agg(["mean", "std"]).round(3).to_string())
    print("    compare with the full-feature run: XGBoost ROC-AUC 0.637, PR-AUC 0.561")

# ---- Check 3: the paper's success target (ball regained within 5 s) ---------------------
if RUN_REGAIN:
    print("\n[3] target = y_regain")
    cp.run("y_regain", tag="regain_check")
