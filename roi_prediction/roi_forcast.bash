export MAPS="/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk"
export PID="sub-002S0619"
export DG="cam_${PID}_4tp"

python - <<'PY'
# ============================================================
# ROI-level Attribution Forecasting (0/6/12 -> predict 24)
#   - HARD-CODED SAFE ENV (no None)
#   - READ-ONLY on existing ROI TSVs
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------
# Environment (assert non-None)
# --------------------
MAPS = os.environ.get("MAPS")
PID  = os.environ.get("PID")
DG   = os.environ.get("DG")

assert MAPS is not None, "[ERR] MAPS is None"
assert PID  is not None, "[ERR] PID is None"
assert DG   is not None, "[ERR] DG is None"

MAPS = Path(MAPS).expanduser().resolve()

print(f"[ENV] MAPS={MAPS}")
print(f"[ENV] PID={PID}")
print(f"[ENV] DG={DG}")

# --------------------
# Locate checkpoint
# --------------------
roots = [
    p for p in [
        MAPS/'split-0'/'best-loss',
        MAPS/'split-0'/'best-F1_score'
    ] if p.exists()
]
assert roots, "[ERR] 找不到 split-0/best-*"
ROOT = roots[0]

# --------------------
# Directories
# --------------------
ROI_DIR = ROOT/DG/"roi_gradcam"
OUTD = ROOT/DG/"roi_prediction"
OUTD.mkdir(parents=True, exist_ok=True)

# --------------------
# Load ROI attribution TSVs
# --------------------
files = [
    ROI_DIR/"roi_cam_values_all.tsv",
    ROI_DIR/"roi_cam_values_all_subcortical.tsv"
]

dfs = []
for f in files:
    if f.exists():
        print(f"[LOAD] {f}")
        dfs.append(pd.read_csv(f, sep="\t"))

assert dfs, "[ERR] roi_gradcam 目录下没有可用 TSV"

df = pd.concat(dfs, ignore_index=True)

# --------------------
# Session -> month
# --------------------
def ses_to_month(s):
    return int(s.replace("ses-m",""))

df["month"] = df["session"].apply(ses_to_month)

train_df = df[df["month"].isin([0, 6, 12])].copy()
assert not train_df.empty, "[ERR] 没有 0/6/12 月的数据"

train_df.to_csv(OUTD/"input_used.tsv", sep="\t", index=False)

# --------------------
# Linear extrapolation
# --------------------
rows = []

for roi, g in train_df.groupby("roi_name"):
    g = g.sort_values("month")
    x = g["month"].to_numpy(dtype=float)
    y = g["cam_energy"].to_numpy(dtype=float)

    if len(x) < 2:
        continue

    a, b = np.polyfit(x, y, 1)
    y24 = max(a * 24.0 + b, 0.0)

    rows.append({
        "subject": PID,
        "roi_name": roi,
        "predicted_month": 24,
        "predicted_cam_energy": float(y24),
        "slope_per_month": float(a),
        "intercept": float(b)
    })

pred_df = pd.DataFrame(rows).sort_values(
    "predicted_cam_energy", ascending=False
)

out_pred = OUTD/"pred_month24.tsv"
pred_df.to_csv(out_pred, sep="\t", index=False)

print("\n=== ROI attribution prediction DONE ===")
print(f"[OK] input used : {OUTD/'input_used.tsv'}")
print(f"[OK] prediction : {out_pred}")
print(f"[INFO] n_rois predicted = {len(pred_df)}")
PY
