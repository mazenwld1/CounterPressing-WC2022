"""
Prints raw events around a few turnovers to debug the y_regain label.
Run from the repository root:  python diagnostics/debug_regain.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import counterpress_wc2022 as cp

pd_opts = dict(index=False)
df = cp.build_dataset()
print("overall y_regain rate:", round(df["y_regain"].mean(), 4), "| y_cp rate:", round(df["y_cp"].mean(), 3))

sample = df[df["y_cp"] == 1].sample(4, random_state=1)
for _, r in sample.iterrows():
    ev, _ = cp.load_match(int(r["match_id"]))
    ev["t"] = ev["timestamp"].map(cp.ts)
    p = int(r["possession"])
    w = ev[ev["possession"].between(p - 1, p + 2)][
        ["index", "t", "type", "team", "possession", "possession_team", "play_pattern", "counterpress"]]
    print(f"\n=== match {int(r['match_id'])} | p_next={p} | y_regain={int(r['y_regain'])} ===")
    print(w.head(30).to_string(**pd_opts))
