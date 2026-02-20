python - <<'PY'
# ============================================================
# ROI-level Grad-CAM (READ-ONLY, FIXED)
#   - Uses resampled CAM
#   - Aggregates CAM energy per anatomical ROI
#   - Writes ONLY to new directory
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

# checkpoint root（与你之前逻辑一致）
roots = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
assert roots, "[ERR] 找不到 split-0/best-*"
ROOT = roots[0]

CAM_RS = ROOT/DG/"interpret-gradcam_resampled"

# output dir (NEW ONLY)
OUTD = ROOT/DG/"roi_gradcam"
OUTD.mkdir(parents=True, exist_ok=True)

# --------------------
# Load atlas (Harvard-Oxford)
# --------------------
atlas = datasets.fetch_atlas_harvard_oxford(
    atlas_name="cort-maxprob-thr25-2mm"
)

atlas_img = atlas.maps     # ✅ 已经是 Nifti1Image
labels = atlas.labels

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

# --------------------
# Main loop
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

    # Resample atlas into subject T1 space (nearest-neighbour!)
    atlas_r = resample_to_img(
        atlas_img,
        Timg,
        interpolation="nearest"
    )
    A = atlas_r.get_fdata().astype(int)

    # CAM data
    C = np.squeeze(Cimg.get_fdata()).astype(np.float32)
    C = np.clip(C, 0, None)   # positive CAM only

    for roi_id, roi_name in enumerate(labels):
        if roi_id == 0 or roi_name.lower() == "background":
            continue

        mask = (A == roi_id)
        nvox = int(mask.sum())
        if nvox == 0:
            continue

        energy = float(C[mask].sum())
        mean   = float(C[mask].mean())

        rows.append({
            "subject": PID,
            "session": ses,
            "roi_id": roi_id,
            "roi_name": roi_name,
            "cam_energy": energy,
            "cam_mean": mean,
            "n_voxels": nvox
        })

# --------------------
# Save TSV
# --------------------
df = pd.DataFrame(rows)
df.sort_values(["roi_name","session"], inplace=True)

out_tsv = OUTD/"roi_cam_values.tsv"
df.to_csv(out_tsv, sep="\t", index=False)

# --------------------
# Z-score per ROI (optional)
# --------------------
df_z = df.copy()
df_z["cam_energy_z"] = (
    df.groupby("roi_name")["cam_energy"]
      .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
)

out_z = OUTD/"roi_cam_values_zscore.tsv"
df_z.to_csv(out_z, sep="\t", index=False)

# --------------------
# Human-readable summary
# --------------------
top = (
    df.groupby("roi_name")["cam_energy"]
      .mean()
      .sort_values(ascending=False)
      .head(10)
)

with (OUTD/"roi_cam_summary.txt").open("w") as f:
    f.write(f"Subject: {PID}\n\nTop ROIs by mean Grad-CAM energy:\n")
    for r,v in top.items():
        f.write(f"  {r}: {v:.4f}\n")

print("\n=== ROI-level Grad-CAM DONE ===")
print(f"[OK] {out_tsv}")
print(f"[OK] {out_z}")
print(f"[OK] roi_cam_summary.txt")
PY
