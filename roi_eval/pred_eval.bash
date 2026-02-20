# ====== 与之前一致的环境 ======
export MAPS="/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk"
export PID="sub-002S0619"
export DG="cam_${PID}_4tp"

python - <<'PY'
# ============================================================
# Evaluation: Predicted vs Observed ROI Attribution @ Month 24
#   - Generates Figure (PNG)
#   - READ-ONLY on all existing results
#   - Writes ONLY new evaluation outputs
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# --------------------
# Environment
# --------------------
MAPS = Path(os.environ["MAPS"])
PID  = os.environ["PID"]
DG   = os.environ["DG"]

# checkpoint root
roots = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
assert roots, "[ERR] 找不到 split-0/best-*"
ROOT = roots[0]

ROI_DIR = ROOT/DG/"roi_gradcam"
PRED_DIR = ROOT/DG/"roi_prediction"
OUTD = ROOT/DG/"roi_evaluation"
OUTD.mkdir(parents=True, exist_ok=True)

# --------------------
# Load prediction
# --------------------
pred = pd.read_csv(PRED_DIR/"pred_month24.tsv", sep="\t")

# --------------------
# Load observed month-24 attribution
# --------------------
files = [
    ROI_DIR/"roi_cam_values_all.tsv",
    ROI_DIR/"roi_cam_values_all_subcortical.tsv"
]

dfs = []
for f in files:
    if f.exists():
        dfs.append(pd.read_csv(f, sep="\t"))

assert dfs, "[ERR] 没有 ROI attribution TSV"
df = pd.concat(dfs, ignore_index=True)

obs = df[df["session"] == "ses-m024"].copy()
obs = obs[["roi_name", "cam_energy"]].rename(
    columns={"cam_energy": "observed_cam_energy"}
)

# --------------------
# Align prediction & observation
# --------------------
merged = pred.merge(obs, on="roi_name", how="inner")

assert not merged.empty, "[ERR] 预测与真实 ROI 无交集"

# --------------------
# Spearman rank correlation
# --------------------
rho, pval = spearmanr(
    merged["observed_cam_energy"],
    merged["predicted_cam_energy"]
)

# --------------------
# Plot: Predicted vs Observed
# --------------------
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    merged["observed_cam_energy"],
    merged["predicted_cam_energy"],
    alpha=0.6,
    s=35,
    color="tab:blue"
)

# identity line
lim = max(
    merged["observed_cam_energy"].max(),
    merged["predicted_cam_energy"].max()
)
ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1)

# highlight hippocampus / amygdala
for _, r in merged.iterrows():
    name = r["roi_name"].lower()
    if ("hippocampus" in name) or ("amygdala" in name):
        ax.scatter(
            r["observed_cam_energy"],
            r["predicted_cam_energy"],
            color="tab:red",
            s=70,
            edgecolor="black",
            zorder=5
        )
        ax.text(
            r["observed_cam_energy"],
            r["predicted_cam_energy"],
            r["roi_name"].replace("Left ", "L ").replace("Right ", "R "),
            fontsize=8
        )

ax.set_xlabel("Observed ROI attribution (Month 24)")
ax.set_ylabel("Predicted ROI attribution (Month 24)")
ax.set_title(
    f"{PID}\nPredicted vs Observed ROI Attribution @ Month 24\n"
    f"Spearman ρ = {rho:.2f} (p={pval:.1e})"
)

ax.grid(True, alpha=0.3)

out_fig = OUTD/"pred_vs_obs_month24.png"
fig.savefig(out_fig, dpi=200, bbox_inches="tight")
plt.close(fig)

# --------------------
# Save numeric evaluation
# --------------------
merged.to_csv(OUTD/"pred_vs_obs_month24.tsv", sep="\t", index=False)

with open(OUTD/"spearman_summary.txt", "w") as f:
    f.write(f"Subject: {PID}\n")
    f.write(f"Spearman rho: {rho:.4f}\n")
    f.write(f"P-value: {pval:.4e}\n")

print("\n=== Evaluation DONE ===")
print(f"[OK] Figure: {out_fig}")
print(f"[OK] Table : {OUTD/'pred_vs_obs_month24.tsv'}")
print(f"[OK] Spearman: {OUTD/'spearman_summary.txt'}")
PY
