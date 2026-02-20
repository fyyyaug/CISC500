# ====== 与之前完全一致的环境 ======
export MAPS="/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk"
export PID="sub-002S0619"
export DG="cam_${PID}_4tp"

python - <<'PY'
# ============================================================
# Zoomed-in evaluation plot (low-attribution ROIs)
#   - Focus on lower-left corner of Figure 2
#   - Highlights hippocampus / amygdala
#   - READ-ONLY, writes new PNG only
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------
# Environment
# --------------------
MAPS = Path(os.environ["MAPS"])
PID  = os.environ["PID"]
DG   = os.environ["DG"]

roots = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
ROOT = roots[0]

EVAL_DIR = ROOT/DG/"roi_evaluation"
OUTD = EVAL_DIR
OUTD.mkdir(parents=True, exist_ok=True)

# --------------------
# Load merged evaluation table
# --------------------
df = pd.read_csv(EVAL_DIR/"pred_vs_obs_month24.tsv", sep="\t")

# --------------------
# Define zoom threshold
#   Use percentile-based threshold (robust)
# --------------------
q = 0.85   # keep bottom 85% (i.e., zoom into low region)
thr_obs = df["observed_cam_energy"].quantile(q)
thr_pred = df["predicted_cam_energy"].quantile(q)

zoom = df[
    (df["observed_cam_energy"] <= thr_obs) &
    (df["predicted_cam_energy"] <= thr_pred)
].copy()

# --------------------
# Plot zoomed scatter
# --------------------
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    zoom["observed_cam_energy"],
    zoom["predicted_cam_energy"],
    alpha=0.7,
    s=40,
    color="tab:blue"
)

# identity line
lim = max(
    zoom["observed_cam_energy"].max(),
    zoom["predicted_cam_energy"].max()
)
ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)

# highlight hippocampus / amygdala
for _, r in zoom.iterrows():
    name = r["roi_name"].lower()
    if ("hippocampus" in name) or ("amygdala" in name):
        ax.scatter(
            r["observed_cam_energy"],
            r["predicted_cam_energy"],
            color="tab:red",
            s=80,
            edgecolor="black",
            zorder=5
        )
        ax.text(
            r["observed_cam_energy"],
            r["predicted_cam_energy"],
            r["roi_name"].replace("Left ", "L ").replace("Right ", "R "),
            fontsize=9
        )

ax.set_xlabel("Observed ROI attribution (Month 24)")
ax.set_ylabel("Predicted ROI attribution (Month 24)")
ax.set_title(
    f"{PID}\nZoomed view: low-attribution ROIs"
)

ax.grid(True, alpha=0.3)

out_fig = OUTD/"pred_vs_obs_month24_zoom_low.png"
fig.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close(fig)

print("\n=== Zoomed-in evaluation figure DONE ===")
print(f"[OK] {out_fig}")
print(f"[INFO] zoom threshold (obs)  <= {thr_obs:.3e}")
print(f"[INFO] zoom threshold (pred) <= {thr_pred:.3e}")
print(f"[INFO] n_points in zoom = {len(zoom)}")
PY
