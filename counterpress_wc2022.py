"""
Counterpressing detection on StatsBomb open data (FIFA World Cup 2022)
Adapted from Bauer & Anzer (2021), Data Mining and Knowledge Discovery 35:2009-2049.

Usage (from the repository root):
    python counterpress_wc2022.py --quick                    # smoke test
    python counterpress_wc2022.py                            # full experiments
    python counterpress_wc2022.py --data-dir path/to/open-data/data

Key differences from the paper
- No tracking data  -> features come from StatsBomb 360 freeze-frames (only the
  players visible in the camera frame, at ONE instant: the turnover event).
- No expert labels  -> proxy label = StatsBomb's own `counterpress` flag.
- Units are StatsBomb yards (pitch 120 x 80), not metres.
"""
import os
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import ConvexHull
from sklearn.metrics import (roc_auc_score, average_precision_score, matthews_corrcoef,
                             balanced_accuracy_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import json
from pathlib import Path

warnings.filterwarnings("ignore")
COMP, SEASON = 43, 106          # FIFA World Cup 2022
DATA_DIR = Path(os.environ.get("STATSBOMB_DATA", "open-data/data"))   # clone of github.com/statsbomb/open-data; override with --data-dir
RESULTS_DIR = Path("results")   # figures and CSVs are written here
USE_WIN_TYPE = False            # win_type_Carry is a near-perfect shortcut (98% positive) -> excluded
CV_FOLDS, CV_REPEATS, CV_TRIALS = 5, 3, 15   # repeated grouped CV, Optuna trials per outer fold
DROP_NO_FRAME = True            # drop turnovers without a 360 freeze-frame (fair baseline/model comparison)
WINDOW = 5.0                     # seconds: paper's "success" / StatsBomb counterpress window
IN_PLAY = ("Regular Play", "From Counter")   # paper excludes possessions starting with a set piece
LOSS_TYPES = {"Pass", "Carry", "Dribble", "Miscontrol", "Dispossessed", "Clearance"}
REGAIN_TYPES = {"Ball Recovery", "Interception", "Pass", "Carry", "Dribble", "Shot"}
WIN_TYPES = {"Ball Recovery", "Interception", "Duel", "Pass", "Carry", "Ball Receipt*",
             "Clearance", "Dribble", "Miscontrol"}
CACHE = "counterpress_dataset.csv"


def load_match(mid):
    """Read one match from the local open-data folder (events + 360 freeze-frames)."""
    raw = json.load(open(DATA_DIR / "events" / f"{mid}.json", encoding="utf-8"))
    ev = pd.DataFrame([{
        "id": e["id"], "index": e["index"], "period": e["period"],
        "timestamp": e["timestamp"], "minute": e["minute"],
        "type": e["type"]["name"], "possession": e["possession"],
        "possession_team": e["possession_team"]["name"],
        "play_pattern": e["play_pattern"]["name"], "team": e["team"]["name"],
        "location": e.get("location"),
        "under_pressure": e.get("under_pressure", False),
        "counterpress": e.get("counterpress", False),
        "position": (e.get("position") or {}).get("name"),
    } for e in raw])
    f360 = json.load(open(DATA_DIR / "three-sixty" / f"{mid}.json", encoding="utf-8"))
    fr_by_id = {d["event_uuid"]: pd.DataFrame(d["freeze_frame"]) for d in f360}
    return ev, fr_by_id


def ts(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# --------------------------------------------------------------------------
# 1. 360 freeze-frame features (everything from the counterpressing team's view)
# --------------------------------------------------------------------------
def frame_feats(fr, ball):
    """A = team that lost the ball (would counterpress). B = team that won it.
    Coordinates are flipped so A always attacks toward +x."""
    f = {}
    fr = fr[~fr["keeper"]]                       # paper excludes goalkeepers
    if len(fr) == 0:
        return f
    pts = np.array(fr["location"].tolist(), dtype=float)
    pts[:, 0] = 120 - pts[:, 0]
    pts[:, 1] = 80 - pts[:, 1]
    is_b = fr["teammate"].values.astype(bool)    # teammate of the actor (ball winner)
    A, B = pts[~is_b], pts[is_b]
    f["vis_A"], f["vis_B"] = len(A), len(B)

    for name, P in (("A", A), ("B", B)):
        d = np.hypot(*(P - ball).T) if len(P) else np.array([])
        for r in (5, 10, 20, 30):
            f[f"n{name}_{r}"] = int((d <= r).sum())
        f[f"front_{name}"] = int((P[:, 0] > ball[0]).sum()) if len(P) else np.nan
        f[f"behind_{name}"] = int((P[:, 0] <= ball[0]).sum()) if len(P) else np.nan
        if len(P):
            f[f"cx_{name}"], f[f"cy_{name}"] = P.mean(axis=0)
            f[f"std_x_{name}"], f[f"std_y_{name}"] = P.std(axis=0)
        if name == "A" and len(d):
            f["A_min_dist"] = d.min()
            order = np.argsort(d)
            for k in (3, 4, 5):                  # local stretch index (paper Table 1)
                if len(P) >= k:
                    close = P[order[:k]]
                    f[f"A_local{k}_stretch"] = np.hypot(*(close - close.mean(0)).T).mean()
        if len(P) >= 3:                          # effective playing space
            try:
                f[f"hull_{name}"] = ConvexHull(P).volume
            except Exception:
                pass
    f["superiority_10"] = f["nA_10"] - f["nB_10"]   # RQ3 rule of thumb
    return f


# --------------------------------------------------------------------------
# 2. Turnover detection + labels for one match
# --------------------------------------------------------------------------
def match_turnovers(mid):
    ev, fr_by_id = load_match(mid)
    ev["t"] = ev["timestamp"].map(ts)
    if "counterpress" not in ev:
        ev["counterpress"] = np.nan
    ev["counterpress"] = ev["counterpress"].fillna(False).astype(bool)
    ev["under_pressure"] = ev.get("under_pressure", pd.Series(False, index=ev.index)).fillna(False)

    rows, pids = [], sorted(ev["possession"].unique())
    for p_prev, p_next in zip(pids[:-1], pids[1:]):
        a, b = ev[ev.possession == p_prev], ev[ev.possession == p_next]
        ta, tb = a.possession_team.iloc[0], b.possession_team.iloc[0]
        if ta == tb or a.period.iloc[0] != b.period.iloc[0]:
            continue
        if b.play_pattern.iloc[0] not in IN_PLAY:            # exclusion criterion 1
            continue

        a_loss = a[(a.team == ta) & a.type.isin(LOSS_TYPES)]
        b_win = b[(b.team == tb) & b.type.isin(WIN_TYPES) & b.location.notna()]
        if a_loss.empty or b_win.empty:
            continue
        last, first = a_loss.iloc[-1], b_win.iloc[0]

        bx, by = first["location"]                # ball position at the turnover (B's coords)
        x, y = 120 - bx, 80 - by                  # A's perspective
        if x < 60:                                # exclusion criterion 2: loss in own half
            continue
        t0 = first["t"]

        # ---- labels -------------------------------------------------------
        win = ev[(ev.period == first.period) & (ev.t >= t0) & (ev.t <= t0 + WINDOW)]
        y_cp = int(((win.team == ta) & win.counterpress).any())
        # StatsBomb possession ids are "sticky" (a quick contested regain often keeps the old id),
        # so a regain = the losing team A performs an on-ball action within the window.
        y_regain = int(((win.team == ta) & (win.t > t0) & win.type.isin(REGAIN_TYPES)).any())

        # ---- event-based features (only info available AT the turnover) ---
        row = dict(
            match_id=mid, possession=p_next, y_cp=y_cp, y_regain=y_regain,
            x=x, y=y, dist_goal=np.hypot(120 - x, 40 - y), dist_sideline=min(y, 80 - y),
            period=first.period, minute=first.minute,
            loss_type=last["type"], loss_under_pressure=int(bool(last["under_pressure"])),
            win_type=first["type"], prev_play_pattern=a.play_pattern.iloc[0],
            prev_poss_duration=a.t.max() - a.t.min(), prev_poss_events=len(a),
            prev_poss_passes=int((a.type == "Pass").sum()),
            gk_win=int(first.get("position") == "Goalkeeper"),
        )
        if first["id"] in fr_by_id:
            row.update(frame_feats(fr_by_id[first["id"]], np.array([x, y])))
        rows.append(row)
    return rows


def build_dataset():
    if os.path.exists(CACHE):
        cached = pd.read_csv(CACHE)
        if "gk_win" in cached.columns:
            return cached
        print("cached dataset is from an older version of the script -> rebuilding")
    match_ids = [m["match_id"] for m in
                 json.load(open(DATA_DIR / "matches" / str(COMP) / f"{SEASON}.json", encoding="utf-8"))]
    rows = []
    for i, mid in enumerate(match_ids, 1):
        print(f"match {i}/{len(match_ids)}  ({mid})")
        rows += match_turnovers(mid)
    df = pd.DataFrame(rows)
    df.to_csv(CACHE, index=False)
    return df


# --------------------------------------------------------------------------
# 3. Modelling (mirrors paper: XGBoost + TPE tuning + class-balanced variant)
# --------------------------------------------------------------------------
def tune(X, y, groups, n_trials=40):
    import optuna                                    # TPE sampler is Optuna's default
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            subsample=trial.suggest_float("subsample", 0.3, 1.0),
            min_child_weight=trial.suggest_float("min_child_weight", 0.5, 10),
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
        )
        aucs = []
        for tr, va in GroupKFold(5).split(X, y, groups):   # folds split BY MATCH
            m = xgb.XGBClassifier(**params, eval_metric="logloss")
            m.fit(X.iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
        return np.mean(aucs)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def metrics(y, score, thr=0.5, ranked=True):
    """Threshold-free (ROC-AUC, PR-AUC), fixed-threshold (MCC, balanced accuracy) and top-decile
    (precision@10%, lift@10%) metrics. F1 is deliberately NOT used: with ~50% positives,
    'predict everything' already gets F1 ~0.67."""
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= thr
    out = dict(roc_auc=roc_auc_score(y, score), pr_auc=average_precision_score(y, score),
               mcc=matthews_corrcoef(y, pred), bal_acc=balanced_accuracy_score(y, pred))
    if ranked:        # not meaningful for the binary naive rule (all scores tie)
        k = max(1, int(round(0.10 * len(y))))
        top = np.argsort(-score, kind="stable")[:k]
        out["p_at_10"] = float(y[top].mean())
        out["lift_10"] = out["p_at_10"] / float(y.mean())
    else:
        out["p_at_10"] = out["lift_10"] = np.nan
    return out


def grouped_folds(groups, n_splits, rep, seed=42):
    """Shuffle whole matches into folds; different shuffle per repeat."""
    rng = np.random.RandomState(seed + rep)
    uniq = groups.unique().copy()
    rng.shuffle(uniq)
    fold_of = {g: i % n_splits for i, g in enumerate(uniq)}
    f = groups.map(fold_of).values
    for k in range(n_splits):
        yield np.where(f != k)[0], np.where(f == k)[0]


def nested_cv(X, y, groups):
    """Outer: repeated grouped CV over all matches. Inner: Optuna tuning on the outer-train part only."""
    rows = []
    for rep in range(CV_REPEATS):
        for k, (tr, te) in enumerate(grouped_folds(groups, CV_FOLDS, rep)):
            Xtr, Xte, ytr, yte, gtr = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr]
            best = tune(Xtr, ytr, gtr, n_trials=CV_TRIALS)
            spw = (ytr == 0).sum() / (ytr == 1).sum()
            m1 = xgb.XGBClassifier(**best, eval_metric="logloss").fit(Xtr, ytr)
            m2 = xgb.XGBClassifier(**best, scale_pos_weight=spw, eval_metric="logloss").fit(Xtr, ytr)
            naive = Xte["nA_5"].fillna(0).clip(0, 1).values.astype(float)
            for name, sc, ranked in (("Naive rule (A player <=5 yd)", naive, False),
                                     ("XGBoost", m1.predict_proba(Xte)[:, 1], True),
                                     ("XGBoost + class balancer", m2.predict_proba(Xte)[:, 1], True)):
                rows.append(dict(model=name, rep=rep, fold=k, base_rate=yte.mean(),
                                 **metrics(yte, sc, ranked=ranked)))
            print(f"  rep {rep + 1}/{CV_REPEATS} fold {k + 1}/{CV_FOLDS} done")
    return pd.DataFrame(rows)


def run(target="y_cp", exclude_gk=False, drop_y=False, tag=None):
    """One full experiment: data prep -> nested grouped CV -> SHAP figure. Returns the per-fold results."""
    tag = tag or target
    print(f"\n{'=' * 78}\nRUN '{tag}': target={target} exclude_gk={exclude_gk} drop_raw_y={drop_y}\n{'=' * 78}")
    df = build_dataset()
    print(df.shape, "| turnovers without a 360 frame:", df["nA_10"].isna().mean().round(3))
    if DROP_NO_FRAME:
        df = df[df["nA_10"].notna()].reset_index(drop=True)
    print("with frame:", df.shape, "| positive rate:", df[target].mean().round(3))

    print("\nturnovers won by the goalkeeper (gk_win=1):")
    print(df.groupby("gk_win")[target].agg(["mean", "count"]).round(3).to_string())
    if exclude_gk:
        df = df[df["gk_win"] == 0].reset_index(drop=True)
        print("after excluding goalkeeper wins:", df.shape, "| positive rate:", df[target].mean().round(3))

    drop = ["match_id", "possession", "y_cp", "y_regain", "gk_win"]
    drop += [] if USE_WIN_TYPE else ["win_type"]
    drop += ["y"] if drop_y else []
    cats = [c for c in ["loss_type", "win_type", "prev_play_pattern"] if c in df.columns and c not in drop]
    X = pd.get_dummies(df.drop(columns=drop), columns=cats)
    y, groups = df[target], df["match_id"]
    print(f"{X.shape[1]} features | {len(df)} turnovers | {groups.nunique()} matches")

    res = nested_cv(X, y, groups)
    res.insert(0, "run", tag)
    cols = ["roc_auc", "pr_auc", "mcc", "bal_acc", "p_at_10", "lift_10"]
    print(f"\n--- {tag}: repeated grouped CV, mean (std) over {res.groupby(['rep', 'fold']).ngroups} folds "
          f"| base rate {res.base_rate.mean():.3f} ---")
    summ = res.groupby("model")[cols].agg(["mean", "std"])
    print(summ.round(3).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(RESULTS_DIR / f"cv_{tag}.csv", index=False)      # saved first, so a SHAP problem cannot lose it

    # final model on ALL matches -> interpretation only (SHAP), never quoted as performance
    try:
        best = tune(X, y, groups, n_trials=40 if CV_TRIALS >= 15 else CV_TRIALS)
        final = xgb.XGBClassifier(**best, eval_metric="logloss").fit(X, y)
        import shap
        import matplotlib.pyplot as plt
        sv = shap.TreeExplainer(final).shap_values(X)
        plt.figure()
        shap.summary_plot(sv, X, max_display=15, show=False)
        plt.title(f"SHAP - {tag}")
        plt.savefig(RESULTS_DIR / f"shap_{tag}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"saved {RESULTS_DIR / f'shap_{tag}.png'}")
    except Exception as e:      # e.g. shap/pyspark import problems on Windows
        print(f"\n[warning] SHAP figure skipped for '{tag}': {type(e).__name__}: {e}")
        print("          The cross-validation results above are unaffected. See README > Troubleshooting.")
    return res


RUNS = {
    "cp":        dict(target="y_cp"),                                           # StatsBomb counterpress flag
    "regain":    dict(target="y_regain"),                                       # ball regained within 5 s
    "cp_strict": dict(target="y_cp", exclude_gk=True, drop_y=True),             # no GK wins, no raw y
    "regain_strict": dict(target="y_regain", exclude_gk=True, drop_y=True),    # same filters for the regain target
}
DEFAULT_RUNS = ["cp", "regain", "cp_strict"]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="1 CV repeat, 5 Optuna trials (smoke test)")
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS), help="comma-separated subset of: " + ", ".join(RUNS))
    ap.add_argument("--data-dir", default=None, help="path to the StatsBomb open-data/data folder")
    args = ap.parse_args()
    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
    if not (DATA_DIR / "matches" / str(COMP) / f"{SEASON}.json").exists():
        raise SystemExit(f"StatsBomb data not found at {DATA_DIR.resolve()}\n"
                         "Clone https://github.com/statsbomb/open-data into ./open-data or pass --data-dir")
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.quick:
        CV_REPEATS, CV_TRIALS = 1, 5
    all_res = pd.concat([run(tag=t, **RUNS[t]) for t in args.runs.split(",")], ignore_index=True)
    out_csv = RESULTS_DIR / "final_results.csv" if args.runs == ",".join(DEFAULT_RUNS) else RESULTS_DIR / f"final_results_{args.runs.replace(',', '_')}.csv"
    all_res.to_csv(out_csv, index=False)

    cols = ["roc_auc", "pr_auc", "mcc", "bal_acc", "lift_10"]
    g = all_res.groupby(["run", "model"])[cols]
    tab = (g.mean().round(3).astype(str) + " (" + g.std().round(3).astype(str) + ")")
    print("\n" + "=" * 78 + "\nFINAL SUMMARY: mean (std) over grouped-CV folds\n" + "=" * 78)
    print(tab.to_string())
    print(f"\nsaved {out_csv} + shap_<run>.png")
