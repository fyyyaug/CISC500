python - <<'PY'
# ============================================================
# ROI-level Grad-CAM (SUBCORTICAL, READ-ONLY)
#   - Harvard-Oxford subcortical atlas (maxprob thr25, 2mm)
#   - Aggregates CAM energy per subcortical ROI
#   - Writes ONLY to new files (no overwrite of cortical outputs)
# ============================================================

import os, glob
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import datasets
from nilearn.image import resample_to_img

# --------------------
# Environment
# --------------------
CAPS = Path(os.environ["CAPS"])
MAPS = Path(os.environ["MAPS"])
PID  = os.environ["PID"]
DG   = os.environ["DG"]
SESSIONS = [s for s in os.environ.get("SESS","").split() if s]

# checkpoint root
roots = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
assert roots, "[ERR] 找不到 split-0/best-*"
ROOT = roots[0]

CAM_RS = ROOT/DG/"interpret-gradcam_resampled"

# output dir (NEW ONLY)
OUTD = ROOT/DG/"roi_gradcam"
OUTD.mkdir(parents=True, exist_ok=True)

# --------------------
# Load atlas (Harvard-Oxford SUBCORTICAL)
# --------------------
atlas = datasets.fetch_atlas_harvard_oxford(
    atlas_name="sub-maxprob-thr25-2mm"
)
atlas_img = atlas.maps      # already Nifti1Image
labels = atlas.labels       # list of ROI names

# --------------------
# Helpers
# --------------------
def find_t1(sub, ses):
    for pat in [
        CAPS/"subjects"/sub/ses/"t1_linear"/f"{sub}_{ses}_*_desc-Crop_res-1x1x1_T1w.nii.gz",
        CAPS/"subjects"/sub/ses/"t1_linear"/f"{sub}_{ses}_*_T1w.nii.gz",
    ]:
        fs = sorted(glob.glob(str(pat)))
        if fs: return fs[0]
    fs = sorted(glob.glob(str(CAPS/"subjects"/sub/ses/"**/*T1w*.nii.gz"), recursive=True))
    return fs[0] if fs else None

def find_cam(sub, ses):
    p = CAM_RS/f"{sub}_{ses}_map_resamp.nii.gz"
    return p if p.exists() else None

def is_background_label(name: str) -> bool:
    n = (name or "").strip().lower()
    return n in {"background", "bg", "unknown"} or n.startswith("background")

# --------------------
# Main computation
# --------------------
rows = []

for ses in SESSIONS:
    t1p = find_t1(PID, ses)
    camp = find_cam(PID, ses)

    if not (t1p and camp):
        print(f"[SKIP] {ses}: missing T1 or CAM")
        continue

    Timg = nib.load(str(t1p))
    Cimg = nib.load(str(camp))

    # Resample atlas into subject T1 space (nearest neighbour)
    atlas_r = resample_to_img(atlas_img, Timg, interpolation="nearest")
    A = atlas_r.get_fdata().astype(int)

    # CAM data
    C = np.squeeze(Cimg.get_fdata()).astype(np.float32)
    C = np.clip(C, 0, None)

    # ROI loop
    for roi_id, roi_name in enumerate(labels):
        # HO atlases typically use 0 for background
        if roi_id == 0 or is_background_label(roi_name):
            continue

        mask = (A == roi_id)
        nvox = int(mask.sum())
        if nvox == 0:
            continue

        rows.append({
            "subject": PID,
            "session": ses,
            "roi_id": roi_id,
            "roi_name": roi_name,
            "cam_energy": float(C[mask].sum()),
            "cam_mean": float(C[mask].mean()),
            "n_voxels": nvox
        })

df = pd.DataFrame(rows)

if df.empty:
    print("\n[WARN] 结果为空：没有任何 subcortical ROI 命中。")
    print("可能原因：atlas 与 T1 覆盖范围不一致 / resampled CAM 或 T1 路径异常。")
    raise SystemExit(0)

# --------------------
# Save ALL subcortical ROI values
# --------------------
df.sort_values(["roi_name","session"], inplace=True)
out_all = OUTD/"roi_cam_values_all_subcortical.tsv"
df.to_csv(out_all, sep="\t", index=False)

# --------------------
# Per-session ROI ranking (subcortical)
# --------------------
rank_rows = []
for ses in df["session"].unique():
    d = df[df.session == ses].copy()
    d["rank_energy"] = d["cam_energy"].rank(ascending=False, method="min")
    for _, r in d.iterrows():
        rank_rows.append({
            "subject": r.subject,
            "session": r.session,
            "roi_name": r.roi_name,
            "cam_energy": r.cam_energy,
            "rank_energy": int(r.rank_energy)
        })

df_rank = pd.DataFrame(rank_rows)
out_rank = OUTD/"roi_cam_rank_per_session_subcortical.tsv"
df_rank.to_csv(out_rank, sep="\t", index=False)

# --------------------
# Summary text (all ROIs sorted)
# --------------------
summary = (
    df.groupby("roi_name")["cam_energy"]
      .mean()
      .sort_values(ascending=False)
)

out_txt = OUTD/"roi_cam_summary_all_subcortical.txt"
with out_txt.open("w") as f:
    f.write(f"Subject: {PID}\n")
    f.write("All SUBCORTICAL ROIs sorted by mean Grad-CAM energy:\n\n")
    for r,v in summary.items():
        f.write(f"{r}: {v:.6f}\n")

# Quick check: hippocampus/amygdala presence
names_l = [n.lower() for n in summary.index.tolist()]
has_hip = any("hippocampus" in n for n in names_l)
has_amy = any("amygdala" in n for n in names_l)

print("\n=== ROI-level Grad-CAM (SUBCORTICAL) DONE ===")
print(f"[OK] {out_all}")
print(f"[OK] {out_rank}")
print(f"[OK] {out_txt}")
print(f"[CHECK] hippocampus in ROI list? {has_hip}")
print(f"[CHECK] amygdala in ROI list?    {has_amy}")
PY
