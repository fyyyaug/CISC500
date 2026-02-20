# ====== 配置（与你之前一致）======
export CAPS=/home/jovyan/cdl/CAPS
export MAPS=/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk
export PID=sub-002S0619
export DG="cam_${PID}_4tp"
export SESS="ses-m000 ses-m006 ses-m012 ses-m024"

# 选 checkpoint（优先 best-loss）
if [ -f "$MAPS/split-0/best-loss/model.pth.tar" ]; then
  CKPTDIR=best-loss; SELMET=loss
elif [ -f "$MAPS/split-0/best-F1_score/model.pth.tar" ]; then
  CKPTDIR=best-F1_score; SELMET=F1_score
else
  echo "[ERR] 没找到 best-* 权重于 $MAPS"; exit 1
fi
echo "[CKPT] $CKPTDIR  [SELMET] $SELMET"

# 4 行 participants.tsv（诊断占位，不影响 CAM）
cat > /tmp/${DG}.tsv <<EOF
participant_id	session_id	diagnosis
$PID	ses-m000	AD
$PID	ses-m006	AD
$PID	ses-m012	AD
$PID	ses-m024	AD
EOF

# ====== 0) 单受试者：时序对齐 + 强度归一（基线锚定到 m000；若无则最早）======
#    输出：$MAPS/split-0/$CKPTDIR/$DG/vis_norm_T1/${PID}_ses-*_T1w_norm.nii.gz
python - <<'PY'
import os, re, json
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform

CAPS = Path(os.environ["CAPS"]).expanduser().resolve()
MAPS = Path(os.environ["MAPS"]).expanduser().resolve()
PID  = os.environ.get("PID","sub-002S0619")
DG   = os.environ.get("DG", f"cam_{PID}_4tp")
CKPTDIR = os.environ.get("CKPTDIR","best-loss")
SESS = [s.strip() for s in os.environ.get("SESS","ses-m000 ses-m006 ses-m012 ses-m024").split() if s.strip()]
OUTD = MAPS/"split-0"/CKPTDIR/DG/"vis_norm_T1"
OUTD.mkdir(parents=True, exist_ok=True)

def nrm(s): m=re.match(r"^ses[-_]?m?(\d{3})$",s,re.I); return f"ses-m{m.group(1)}" if m else s.lower()
SESS=[nrm(s) for s in SESS]

def find_t1(sub, ses):
    p = CAPS/"subjects"/sub/ses/"t1_linear"
    if p.is_dir():
        c=sorted(p.glob(f"{sub}_{ses}_*_T1w.nii.gz"))
        if c: return c[0]
    # 容错搜索
    for pat in [f"{sub}_{ses}_*_desc-Crop_res-1x1x1_T1w.nii.gz","*T1w.nii.gz","*.nii.gz"]:
        c=sorted((CAPS/"subjects"/sub/ses).rglob(pat))
        c=[x for x in c if x.suffix in [".gz",".nii"] and "seg" not in x.name and "proba" not in x.name]
        if c: return c[0]
    return None

def load(p): return nib.as_closest_canonical(nib.load(str(p)))
def same_grid(a,b,atol=1e-3): import numpy as np; return a.shape==b.shape and np.allclose(a.affine,b.affine,atol=atol)

def resample_to(img, ref):
    try:
        from nibabel.processing import resample_from_to
        return resample_from_to(img, ref, order=1)
    except Exception:
        import numpy as np
        T = np.linalg.inv(img.affine) @ ref.affine
        Tin = np.linalg.inv(T); M,off = Tin[:3,:3], Tin[:3,3]
        dat = img.get_fdata(dtype=np.float32)
        out = affine_transform(dat, matrix=M, offset=off, output_shape=ref.shape,
                               order=1, mode="nearest", cval=0.0, prefilter=True)
        return nib.Nifti1Image(out, ref.affine, ref.header)

# 简易脑掩膜（nilearn优先）
try:
    from nilearn.masking import compute_brain_mask
    def brain_mask(img):
        m = compute_brain_mask(img).get_fdata()>0
        return m.astype(bool) if m.sum()>0 else np.zeros(img.shape, bool)
except Exception:
    def brain_mask(img):
        import numpy as np
        d = img.get_fdata().astype(np.float32); d = np.nan_to_num(d)
        nz = d[d!=0]; 
        if nz.size<1000: return np.zeros_like(d, bool)
        p2,p98 = np.percentile(nz,[2,98]); thr = p2+0.1*(p98-p2) if p98>p2 else np.median(nz)
        m = d>thr
        try:
            import scipy.ndimage as ndi
            lab,n = ndi.label(m)
            if n>1:
                sizes = ndi.sum(m, lab, index=np.arange(1,n+1))
                keep = 1+int(np.argmax(sizes)); m = (lab==keep)
        except Exception: pass
        return m.astype(bool)

import numpy as np

# 收集可用会话
pairs=[]
for s in SESS:
    t1 = find_t1(PID, s)
    if t1: pairs.append((s, t1))
if not pairs:
    print(f"[ERR] {PID}: 无 T1W 可归一", flush=True); raise SystemExit(0)

# 基线：优先 m000
ses_list=[s for s,_ in pairs]
baseline = "ses-m000" if "ses-m000" in ses_list else sorted(ses_list, key=lambda x:int(re.search(r"m(\d{3})",x).group(1)))[0]
bimg = load(dict(pairs)[baseline])
mask = brain_mask(bimg)
bdat = bimg.get_fdata().astype(np.float32)
x = bdat[mask].astype(np.float32)
if x.size==0: med,iqr=0.0,1.0
else:
    p25,p50,p75 = np.percentile(x,[25,50,75]); med=float(p50); iqr=float(max(p75-p25,1e-6))

def apply_norm(img):
    arr = img.get_fdata().astype(np.float32)
    z = (arr - med) / iqr
    z = np.clip(z, -5.0, 5.0)
    return (z+5.0)/10.0

ok=0; miss=0
for s, p in pairs:
    try:
        img = load(p)
        if not same_grid(img, bimg):
            img = resample_to(img, bimg)
        out = apply_norm(img)
        outp = OUTD/f"{PID}_{s}_T1w_norm.nii.gz"
        nib.save(nib.Nifti1Image(out.astype(np.float32), bimg.affine, bimg.header), str(outp))
        ok+=1
    except Exception:
        miss+=1

with (OUTD/"baseline_stats.json").open("w") as f:
    json.dump({"subject":PID,"baseline_session":baseline,"median":med,"iqr":iqr}, f, indent=2)

print(f"[NORM] {PID}: baseline={baseline} ok={ok} miss={miss} out={OUTD}", flush=True)
PY

# ====== 1) 若该组未建立，则 predict 一次把样本登记进组（跳过泄漏检查）======
if [ ! -f "$MAPS/groups/$DG/maps.json" ]; then
  clinicadl predict "$MAPS" "$DG" \
    --caps_directory "$CAPS" \
    --participants_tsv "/tmp/${DG}.tsv" \
    --diagnoses AD --diagnoses CN --diagnoses MCI \
    --selection_metrics "$SELMET" \
    --skip_leak_check --batch_size 1 --n_proc 4 --overwrite
fi

# ====== 2) 逐时点 interpret（先 L5，再回退 L4），保存 NIfTI ======
for S in $SESS; do
  printf "participant_id\tsession_id\tdiagnosis\n$PID\t$S\tAD\n" > /tmp/${DG}_${S}.tsv
  echo "[RUN] $PID $S @L5"
  clinicadl interpret "$MAPS" "$DG" gradcam grad-cam \
    --caps_directory "$CAPS" --participants_tsv "/tmp/${DG}_${S}.tsv" \
    --diagnoses AD --diagnoses CN --diagnoses MCI \
    --selection_metrics "$SELMET" \
    --save_individual --save_nifti --batch_size 1 --n_proc 4 --overwrite \
    --level_grad_cam 5 --target_node 0 || \
  ( echo "[RET] $PID $S -> 回退 L4" && \
    clinicadl interpret "$MAPS" "$DG" gradcam grad-cam \
      --caps_directory "$CAPS" --participants_tsv "/tmp/${DG}_${S}.tsv" \
      --diagnoses AD --diagnoses CN --diagnoses MCI \
      --selection_metrics "$SELMET" \
      --save_individual --save_nifti --batch_size 1 --n_proc 4 --overwrite \
      --level_grad_cam 4 --target_node 0 )
done

echo "---- CAM NIfTI ----"
find "$MAPS/split-0/$CKPTDIR/$DG/interpret-gradcam" -maxdepth 1 -type f -name "${PID}_ses-*_map.nii.gz" | sort

# ====== 3) 生成固定世界坐标三视图叠加（优先使用归一后的 T1），并拼图 ======
python - <<'PY'
import os, glob
import numpy as np, nibabel as nib
from pathlib import Path
from scipy.ndimage import gaussian_filter, zoom, label
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

MAPS = Path(os.environ["MAPS"])
CAPS = Path(os.environ["CAPS"])
PID  = os.environ["PID"]
DG   = os.environ["DG"]
SESSIONS = [s for s in os.environ.get("SESS","ses-m000 ses-m006 ses-m012 ses-m024").split() if s]

# checkpoint 根目录
root_candidates = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
assert root_candidates, "[ERR] 没找到 split-0/{best-loss,best-F1_score}"

# 归一 T1 目录
NORMD = root_candidates[0]/DG/'vis_norm_T1'

def to3(a):
    a=np.squeeze(np.asarray(a))
    if a.ndim==3: return a
    if a.ndim==4 and a.shape[-1]==1: return a[...,0]
    if a.ndim==4 and a.shape[0]==1: return a[0]
    return a[...,0]

def resize_to(src, ref_shape):
    if src.shape!=ref_shape:
        fac = (np.array(ref_shape, float)/np.array(src.shape, float))
        src = zoom(src, fac, order=3)
    return src

def find_t1(sub, ses):
    # 1) 先用归一后的
    p = NORMD/f"{sub}_{ses}_T1w_norm.nii.gz"
    if p.exists(): return str(p)
    # 2) 回退原 T1
    for pat in [
        CAPS/"subjects"/sub/ses/"t1_linear"/f"{sub}_{ses}_*_desc-Crop_res-1x1x1_T1w.nii.gz",
        CAPS/"subjects"/sub/ses/"t1_linear"/f"{sub}_{ses}_*_T1w.nii.gz",
    ]:
        fs = sorted(glob.glob(str(pat)))
        if fs: return fs[0]
    fs = sorted(glob.glob(str(CAPS/"subjects"/sub/ses/"**/*T1w*.nii.gz"), recursive=True))
    return fs[0] if fs else None

def find_cam(sub, ses):
    # 搜索顺序：成组目录 -> 回退旧命名
    for root in root_candidates:
        for dg in (DG, "cam49_tp_L5", "cam49_tp_L4"):
            fs = sorted(glob.glob(str(root/dg/"interpret-gradcam"/f"{sub}_{ses}_*_map.nii.gz")))
            if fs: return Path(fs[0]), (root/dg/"interpret-gradcam")
    return None, None

def prep_cam(Timg, cam_path):
    T = np.squeeze(Timg.get_fdata()).astype(float)
    # 背景按 p2-p98 归一，仅用于显示（强度对比）
    mask = T!=0
    nnz = T[mask]; 
    if nnz.size>0:
        p2,p98=np.percentile(nnz,[2,98]); Tn = np.clip((T-p2)/(p98-p2+1e-8),0,1)
    else:
        Tn = (T - T.min())/(T.ptp()+1e-8)
    A = nib.load(str(cam_path)).get_fdata().astype(float)
    A = resize_to(to3(A), T.shape)
    C = np.clip(A,0,None)*mask
    C = gaussian_filter(C, sigma=(1.6,2.6,2.6))
    thr = np.percentile(C[mask], 95.0) if C[mask].size>0 else 0.0
    L,n = label(C>=thr)
    if n>0:
        sizes = [(i,(L==i).sum()) for i in range(1,n+1)]
        keep = max(sizes, key=lambda x:x[1])[0]
        C = np.where(L==keep, C, 0)
        C = gaussian_filter(C, sigma=(0.8,1.2,1.2))
    hi = np.percentile(C[C>0], 99.0) if (C>0).any() else 1.0
    Ov = np.zeros_like(C); 
    if hi>0 and (C>0).any():
        Ov[C>0] = np.clip(C[C>0]/(hi+1e-12), 0, 1)
    return Tn, Ov

# 2) 先确定参考世界坐标（优先 ses-m000）
ref_world=None
for ses in (["ses-m000"]+[s for s in SESSIONS if s!="ses-m000"]):
    cam_path,_ = find_cam(PID, ses); t1p = find_t1(PID, ses)
    if not(cam_path and t1p): continue
    Timg = nib.load(t1p); Tn, Ov = prep_cam(Timg, cam_path)
    if (Ov>0).any():
        i,j,k = np.unravel_index(np.argmax(Ov), Ov.shape)
        xw,yw,zw,_ = (Timg.affine @ np.array([i,j,k,1.0]))
        ref_world=(xw,yw,zw); break
assert ref_world is not None, "[ERR] 没有可用 CAM 来确定参考坐标"

save_root = (root_candidates[0]/DG/"interpret-gradcam"/"vis_fused_fixed")
save_root.mkdir(parents=True, exist_ok=True)

def render_three(Tn, Ov, ijk, title, out_png):
    i,j,k = ijk
    panes=[(Tn[:,:,k], Ov[:,:,k], "Axial"),
           (Tn[:,j,:], Ov[:,j,:], "Coronal"),
           (Tn[i,:,:], Ov[i,:,:], "Sagittal")]
    fig = plt.figure(figsize=(12.5,3.8))
    gs = GridSpec(1,4,width_ratios=[1,1,1,0.05], wspace=0.08)
    im=None
    for col,(bg,ov,lab) in enumerate(panes):
        ax=fig.add_subplot(gs[0,col])
        ax.imshow(np.rot90(bg), cmap="gray", vmin=0, vmax=1)
        im=ax.imshow(np.rot90(ov), cmap="hot", vmin=0, vmax=1, alpha=0.50)
        ax.set_title(f"{title} {lab}"); ax.axis("off")
    cax=fig.add_subplot(gs[0,3]); cb=plt.colorbar(im, cax=cax); cb.set_label("Grad-CAM (0–1)")
    fig.savefig(str(out_png), dpi=200, bbox_inches="tight"); plt.close(fig)

# 3) 按固定世界坐标输出每个时点的三视图
for ses in SESSIONS:
    cam_path,_ = find_cam(PID, ses); t1p = find_t1(PID, ses)
    if not(cam_path and t1p):
        print(f"[SKIP] {ses} 缺 CAM 或 T1"); continue
    Timg = nib.load(t1p); Tn, Ov = prep_cam(Timg, cam_path)
    inv = np.linalg.inv(Timg.affine); xw,yw,zw = ref_world
    i,j,k,_ = (inv @ np.array([xw,yw,zw,1.0]))
    i = int(np.clip(round(i), 0, Tn.shape[0]-1))
    j = int(np.clip(round(j), 0, Tn.shape[1]-1))
    k = int(np.clip(round(k), 0, Tn.shape[2]-1))
    out_png = save_root/f"{PID}_{ses}_fixedSlices.png"
    render_three(Tn, Ov, (i,j,k), f"{PID} {ses}", out_png)
    print(f"[OK] 写出: {out_png}")

# 4) 拼图（行=时点，列=三视图）
big = save_root/f"{PID}_4tp_fixed_compare.png"
fig = plt.figure(figsize=(12, 3*len(SESSIONS)))
gs = GridSpec(len(SESSIONS), 3, wspace=0.05, hspace=0.05)
r=0
for ses in SESSIONS:
    cam_path,_ = find_cam(PID, ses); t1p = find_t1(PID, ses)
    if not(cam_path and t1p): continue
    Timg = nib.load(t1p); Tn, Ov = prep_cam(Timg, cam_path)
    inv = np.linalg.inv(Timg.affine); xw,yw,zw = ref_world
    i,j,k,_ = (inv @ np.array([xw,yw,zw,1.0]))
    i = int(np.clip(round(i), 0, Tn.shape[0]-1))
    j = int(np.clip(round(j), 0, Tn.shape[1]-1))
    k = int(np.clip(round(k), 0, Tn.shape[2]-1))
    panels=[(Tn[:,:,k],Ov[:,:,k]),(Tn[:,j,:],Ov[:,j,:]),(Tn[i,:,:],Ov[i,:,:])]
    for c,(bg,ov) in enumerate(panels):
        ax=fig.add_subplot(gs[r,c])
        ax.imshow(np.rot90(bg), cmap="gray", vmin=0, vmax=1)
        ax.imshow(np.rot90(ov), cmap="hot", vmin=0, vmax=1, alpha=0.50)
        ax.set_axis_off()
    r+=1
fig.savefig(str(big), dpi=200, bbox_inches="tight"); plt.close(fig)
print(f"[OK] 写出: {big}")
PY
