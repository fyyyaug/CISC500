python - <<'PY'
# ============================================================
# ROI-level Grad-CAM (ALL ROIs, READ-ONLY)
#   - Reports ALL ROIs (not just Top-N)
#   - Adds per-session ROI ranking
#   - Writes ONLY to new files
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
# Load atlas (Harvard-Oxford)
# --------------------
atlas = datasets.fetch_atlas_harvard_oxford(
    atlas_name="cort-maxprob-thr25-2mm"
)
atlas_img = atlas.maps          # already Nifti1Image
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

    # resample atlas to subject T1
    atlas_r = resample_to_img(atlas_img, Timg, interpolation="nearest")
    A = atlas_r.get_fdata().astype(int)

    C = np.squeeze(Cimg.get_fdata()).astype(np.float32)
    C = np.clip(C, 0, None)

    for roi_id, roi_name in enumerate(labels):
        if roi_id == 0 or roi_name.lower() == "background":
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

# --------------------
# Save ALL ROI values
# --------------------
df.sort_values(["roi_name","session"], inplace=True)
out_all = OUTD/"roi_cam_values_all.tsv"
df.to_csv(out_all, sep="\t", index=False)

# --------------------
# Per-session ROI ranking
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
out_rank = OUTD/"roi_cam_rank_per_session.tsv"
df_rank.to_csv(out_rank, sep="\t", index=False)

# --------------------
# Optional: summary text (Top-10 still allowed)
# --------------------
summary = (
    df.groupby("roi_name")["cam_energy"]
      .mean()
      .sort_values(ascending=False)
)

with (OUTD/"roi_cam_summary_all.txt").open("w") as f:
    f.write(f"Subject: {PID}\n")
    f.write("All ROIs sorted by mean Grad-CAM energy:\n\n")
    for r,v in summary.items():
        f.write(f"{r}: {v:.6f}\n")

print("\n=== ROI-level Grad-CAM (ALL ROIs) DONE ===")
print(f"[OK] {out_all}")
print(f"[OK] {out_rank}")
print(f"[OK] roi_cam_summary_all.txt")
PY
