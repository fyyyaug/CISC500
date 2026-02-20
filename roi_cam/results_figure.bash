# ============================================================
# Figure 1: Cortical + Subcortical ROI Grad-CAM overview
#   - Averaged across time
#   - NO subject identifier in figure
#   - READ-ONLY
# ============================================================

python - <<'PY'
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------
# Explicit MAPS experiment dir
# --------------------
MAPS_WORKDIR = Path(
    os.environ.get(
        "MAPS_WORKDIR",
        "/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk"
    )
)

DG = os.environ.get("DG", "cam_sub-002S0619_4tp")

# checkpoint root
roots = [
    MAPS_WORKDIR/'split-0'/'best-loss',
    MAPS_WORKDIR/'split-0'/'best-F1_score'
]
roots = [p for p in roots if p.exists()]
assert roots, f"[ERR] No best-* under {MAPS_WORKDIR}/split-0"
ROOT = roots[0]

# input TSVs
ROI_DIR = ROOT/DG/"roi_gradcam"
cort_tsv = ROI_DIR/"roi_cam_values_all.tsv"
sub_tsv  = ROI_DIR/"roi_cam_values_all_subcortical.tsv"

assert cort_tsv.exists(), f"[ERR] Missing {cort_tsv}"
assert sub_tsv.exists(),  f"[ERR] Missing {sub_tsv}"

# output
FIG_OUT = ROI_DIR/"figure_roi_overview_avg_time.png"

# --------------------
# Load data
# --------------------
df_cort = pd.read_csv(cort_tsv, sep="\t")
df_sub  = pd.read_csv(sub_tsv,  sep="\t")

# mean across time (sessions)
cort_mean = (
    df_cort.groupby("roi_name")["cam_energy"]
            .mean()
            .sort_values(ascending=False)
)

sub_mean = (
    df_sub.groupby("roi_name")["cam_energy"]
          .mean()
          .sort_values(ascending=False)
)

# --------------------
# Select ROIs for plotting
# --------------------
TOP_N = 12
cort_plot = cort_mean.head(TOP_N)

# remove non-informative giant regions
drop_keywords = ["cerebral cortex", "white matter"]
sub_plot = sub_mean[
    ~sub_mean.index.str.lower().str.contains("|".join(drop_keywords))
]

# --------------------
# Plot
# --------------------
plt.figure(figsize=(14, 5))

# ---- Cortical panel ----
ax1 = plt.subplot(1, 2, 1)
cort_plot[::-1].plot(kind="barh", ax=ax1, color="#4C72B0")
ax1.set_title("Cortical ROIs")
ax1.set_xlabel("Mean Grad-CAM energy (averaged across time)")
ax1.set_ylabel("")
ax1.grid(axis="x", linestyle="--", alpha=0.4)

# ---- Subcortical panel ----
ax2 = plt.subplot(1, 2, 2)
colors = []
for name in sub_plot.index:
    lname = name.lower()
    if "hippocampus" in lname:
        colors.append("#DD8452")
    elif "amygdala" in lname:
        colors.append("#C44E52")
    else:
        colors.append("#55A868")

sub_plot[::-1].plot(kind="barh", ax=ax2, color=colors)
ax2.set_title("Subcortical ROIs")
ax2.set_xlabel("Mean Grad-CAM energy (averaged across time)")
ax2.set_ylabel("")
ax2.grid(axis="x", linestyle="--", alpha=0.4)

ax2.text(
    0.98, 0.02,
    "Orange: Hippocampus\nRed: Amygdala",
    transform=ax2.transAxes,
    ha="right", va="bottom",
    fontsize=9
)

# Global title (NO subject name)
plt.suptitle(
    "ROI-level Grad-CAM attribution overview\n(Averaged across longitudinal time points)",
    fontsize=14, y=1.02
)

plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
plt.close()

print("\n=== FIGURE 1 GENERATED (AVERAGED ACROSS TIME) ===")
print(f"[OK] {FIG_OUT}")
PY
