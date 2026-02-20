export CAPS=/home/jovyan/cdl/CAPS
export MAPS=/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft__visnolchk
export PID=sub-002S0619
export DG="cam_${PID}_4tp"
export SESS="ses-m000 ses-m006 ses-m012 ses-m024"

python - <<'PY'
import os, glob, warnings
from pathlib import Path
import numpy as np, nibabel as nib
from datetime import datetime
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter, zoom, label, center_of_mass

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

MAPS = Path(os.environ["MAPS"]).expanduser().resolve()
CAPS = Path(os.environ["CAPS"]).expanduser().resolve()
PID  = os.environ["PID"];  DG = os.environ["DG"]
SESSIONS = [s for s in os.environ.get("SESS","ses-m000 ses-m006 ses-m012 ses-m024").split() if s]

# 1) 选择 checkpoint 根目录（优先 best-loss）
roots = [p for p in [MAPS/'split-0'/'best-loss', MAPS/'split-0'/'best-F1_score'] if p.exists()]
assert roots, "[ERR] split-0/{best-loss,best-F1_score} 不存在"
root = roots[0]

# 2) 输出目录（时间戳，确保不覆盖）
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_single = root/DG/"interpret-gradcam"/"vis_template_clone"/ts/"single"
out_fused  = root/DG/"interpret-gradcam"/"vis_template_clone"/ts/"fused"
out_single.mkdir(parents=True, exist_ok=True)
out_fused.mkdir(parents=True, exist_ok=True)

def safe_save(fig, path):
    p = Path(path); stem, suf = p.stem, p.suffix; k=0; q=p
    while q.exists():
        k+=1; q=p.with_name(f"{stem}__{k}{suf}")
    fig.savefig(str(q), dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] 写出: {q}")

# 3) 工具函数（与“你那版正确代码”一致的处理思想）
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
    # 搜索顺序：当前DG → 你之前常用的两个备选
    for dg in (DG, "cam49_tp_L5", "cam49_tp_L4"):
        fs = sorted(glob.glob(str(root/dg/"interpret-gradcam"/f"{sub}_{ses}_*_map.nii.gz")))
        if fs: return Path(fs[0])
    return None

def to3(a):
    a = np.squeeze(np.asarray(a))
    if a.ndim==3: return a
    if a.ndim==4 and a.shape[-1]==1: return a[...,0]
    if a.ndim==4 and a.shape[0]==1:  return a[0]
    return a[...,0]

def resize_like(src, ref_shape):
    # 按“形状比”重采样（与你模板一致）；目标 t1_linear 已是 1mm 等距
    if src.shape!=ref_shape:
        fac = (np.array(ref_shape, float)/np.array(src.shape, float))
        src = zoom(src, fac, order=3)
    return src

def brain_mask_for_stats(Timg):
    # 仅用于统计/遮罩，不抠黑背景
    try:
        from nilearn.masking import compute_brain_mask
        m = compute_brain_mask(Timg).get_fdata()>0
        if m.sum()>0: return m.astype(bool)
    except Exception:
        pass
    d = np.asarray(Timg.get_fdata(), dtype=np.float32)
    nz = d[d!=0]
    if nz.size<2000: return np.zeros_like(d, bool)
    p2,p98 = np.percentile(nz,[2,98]); thr = p2 + 0.1*(p98-p2) if p98>p2 else float(np.median(nz))
    m = d>thr
    L,n = label(m)
    if n>1:
        sizes = [(i,(L==i).sum()) for i in range(1,n+1)]
        keep = max(sizes, key=lambda x:x[1])[0]
        m = (L==keep)
    return m.astype(bool)

def prep_once(Timg, CAMimg):
    # T1 灰度：脑内统计 p2–p98 窗宽（不抠黑），外部保持原样
    T = np.squeeze(Timg.get_fdata()).astype(np.float32)
    M = brain_mask_for_stats(Timg)
    Tn = T.copy()
    if M.sum()>0:
        v = T[M]; p2,p98 = np.percentile(v,[2,98])
        if p98>p2:
            Tn = np.clip((T - p2)/(p98 - p2), 0, 1)
        else:
            Tn = (T - T.min())/(T.ptp()+1e-6)
    else:
        Tn = (T - T.min())/(T.ptp()+1e-6)

    # CAM：≥0 → 乘脑掩膜 → 平滑(1.6,2.6,2.6) → 95分位阈值 → 最大连通块
    A = np.asarray(CAMimg.get_fdata(), dtype=np.float32)
    A = to3(A)
    A = resize_like(A, T.shape)
    C = np.clip(A, 0, None) * M
    if C.max()>0:
        C = gaussian_filter(C, sigma=(1.6,2.6,2.6))
        nz = C[M & (C>0)]
        thr = np.percentile(nz, 95.0) if nz.size>0 else 0.0
        if thr>0:
            keep = (C>=thr)
            L,n = label(keep)
            if n>1:
                sizes = [(i,(L==i).sum()) for i in range(1,n+1)]
                k = max(sizes, key=lambda x:x[1])[0]
                C = np.where(L==k, C, 0.0)
            else:
                C = C * keep
        # 再轻平滑，避免块状边缘
        C = gaussian_filter(C, sigma=(0.8,1.2,1.2))
        # 99th分位归一，得到叠加热图 Ov
        nz = C[C>0]; hi = np.percentile(nz, 99.0) if nz.size>0 else 0.0
        Ov = np.zeros_like(C, dtype=np.float32)
        if hi>0: Ov[C>0] = np.clip(C[C>0]/(hi+1e-12), 0, 1)
    else:
        Ov = C
    return Tn, Ov, M

# 4) 先确定“固定世界坐标”：在 ses-m000 的**最大连通块质心**
ref_world=None; ref_info=None
for ses in (["ses-m000"]+[s for s in SESSIONS if s!="ses-m000"]):
    t1p, camp = find_t1(PID, ses), find_cam(PID, ses)
    if not (t1p and camp): continue
    Timg = nib.as_closest_canonical(nib.load(t1p))
    Aimg = nib.as_closest_canonical(nib.load(str(camp)))
    Tn, Ov, M = prep_once(Timg, Aimg)
    if (Ov>0).any():
        # 只在 Ov>0 的地方计算连通块质心
        mask = Ov>0
        com = center_of_mass(mask.astype(np.float32))
        i,j,k = [int(round(x)) for x in com]
        xw,yw,zw,_ = (Timg.affine @ np.array([i,j,k,1.0]))
        ref_world = (xw,yw,zw); ref_info=(ses,(i,j,k)); break
assert ref_world is not None, "[ERR] 无法从 CAM 确定参考点"

def render_triptych(Tn, Ov, ijk, title):
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
    return fig

# 5) 逐会话渲染（切片锁死在同一世界点）
ijk_of={}
for ses in SESSIONS:
    t1p, camp = find_t1(PID, ses), find_cam(PID, ses)
    if not (t1p and camp):
        print(f"[SKIP] {ses} 缺 T1 或 CAM"); continue
    Timg = nib.as_closest_canonical(nib.load(t1p))
    Aimg = nib.as_closest_canonical(nib.load(str(camp)))
    Tn, Ov, M = prep_once(Timg, Aimg)
    inv = np.linalg.inv(Timg.affine); xw,yw,zw = ref_world
    i,j,k,_ = (inv @ np.array([xw,yw,zw,1.0]))
    i = int(np.clip(round(i), 0, Tn.shape[0]-1))
    j = int(np.clip(round(j), 0, Tn.shape[1]-1))
    k = int(np.clip(round(k), 0, Tn.shape[2]-1))
    ijk_of[ses]=(i,j,k)
    fig = render_triptych(Tn, Ov, (i,j,k), f"{PID} {ses}")
    safe_save(fig, out_single/f"{PID}_{ses}_triptych.png")

# 6) 拼图（共用同一管线与同一坐标）
if ijk_of:
    fig = plt.figure(figsize=(12, 3*len(ijk_of)))
    gs = GridSpec(len(ijk_of), 3, wspace=0.05, hspace=0.05)
    r=0
    for ses in SESSIONS:
        if ses not in ijk_of: continue
        t1p, camp = find_t1(PID, ses), find_cam(PID, ses)
        Timg = nib.as_closest_canonical(nib.load(t1p))
        Aimg = nib.as_closest_canonical(nib.load(str(camp)))
        Tn, Ov, M = prep_once(Timg, Aimg)
        i,j,k = ijk_of[ses]
        panels=[(Tn[:,:,k],Ov[:,:,k]),(Tn[:,j,:],Ov[:,j,:]),(Tn[i,:,:],Ov[i,:,:])]
        for c,(bg,ov) in enumerate(panels):
            ax=fig.add_subplot(gs[r,c])
            ax.imshow(np.rot90(bg), cmap="gray", vmin=0, vmax=1)
            ax.imshow(np.rot90(ov), cmap="hot", vmin=0, vmax=1, alpha=0.50)
            ax.set_axis_off()
        r+=1
    safe_save(fig, out_fused/f"{PID}_4tp_fixed_compare.png")
else:
    print("[WARN] 无可拼接的会话")
PY

# 查看新产物（只在 vis_template_clone/<timestamp>/ 下；不会覆盖旧文件）
find "$MAPS/split-0" -type f -path "*/interpret-gradcam/vis_template_clone/*/*.png" | sort
