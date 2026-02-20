export TEST_FULL_49="/home/jovyan/cdl/TSV/balanced_trainval_excl_test/train/split-0/data.tsv"

: "${CAPS:=/home/jovyan/cdl/CAPS}"
: "${OUT:=/home/jovyan/cdl/MAPS/t1lin_adni3_baseline_e80_lr3e5_ba}"
: "${OUT_FT:=/home/jovyan/cdl/MAPS/t1lin_adni3_balanced_ft}"

if [ -d "$OUT_FT/split-0" ]; then MODEL_PARENT="$OUT_FT"; else MODEL_PARENT="$OUT"; fi

run_predict_eval() {
  local split="$1" group="$2"
  local base="$MODEL_PARENT/split-0/$split"
  if [ ! -d "$base" ]; then echo "[NOTE] 无 $split，跳过"; return 1; fi

  clinicadl predict "$MODEL_PARENT" "$group" \
    --caps_directory "$CAPS" \
    --participants_tsv "$TEST_FULL_49" \
    --diagnoses AD --diagnoses CN --diagnoses MCI \
    --skip_leak_check --batch_size 8 --n_proc 8 --overwrite >/dev/null 2>&1

  PRED=$(find "$base/$group" -maxdepth 1 -type f -name "*prediction*.tsv" | head -n1)
  if [ -z "$PRED" ]; then echo "[ERR] $split 预测文件未找到"; return 1; fi
  echo "[OK] 使用 $split 预测文件：$PRED"


  python - <<PY
import os, pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix, f1_score
pred = r"""$PRED"""
df = pd.read_csv(pred, sep="\t", dtype=str)
y = df["true_label"].astype(int).to_numpy()
yh = df["predicted_label"].astype(int).to_numpy()
cm = confusion_matrix(y, yh, labels=[0,1,2])
rec = [(cm[i,i]/cm[i,:].sum()) if cm[i,:].sum()>0 else 0 for i in range(3)]
acc = (y==yh).mean(); mba = float(np.mean(rec)); mf1 = f1_score(y, yh, average="macro")
print(f"[EVAL $split] Acc={acc:.3f}  Macro-BA={mba:.3f}  Macro-F1={mf1:.3f}  N={len(df)}")
print("Confusion (rows=true, cols=pred, 0/1/2=AD/CN/MCI):")
print(cm)
# 把路径写到一个小文件，供后续纠偏使用
open("/tmp/__last_pred.tsv","w").write(pred)
PY
  return 0
}

GROUP="eval_balanced_49_trainset"
run_predict_eval "best-accuracy" "$GROUP" || run_predict_eval "best-loss" "$GROUP"

# ====== 定向纠偏（不重训）：只抬高 AD/CN 的 logit，自动寻优 ======
if [ -f /tmp/__last_pred.tsv ]; then
  PRED=$(cat /tmp/__last_pred.tsv)
  python - <<PY
import os, pandas as pd, numpy as np
from sklearn.metrics import confusion_matrix, f1_score

pred = r"""$PRED"""
df = pd.read_csv(pred, sep="\t", dtype=str)

need = {"proba0","proba1","proba2","true_label"}
if not need.issubset(df.columns):
    print("[NOTE] 预测文件缺少概率列，跳过纠偏"); raise SystemExit(0)

P = df[["proba0","proba1","proba2"]].astype(float).to_numpy()
P = np.clip(P, 1e-12, 1-1e-12); P /= P.sum(1, keepdims=True)
logP = np.log(P)
y = df["true_label"].astype(int).to_numpy()

def eval_probs(Q):
    pred = Q.argmax(1)
    cm = confusion_matrix(y, pred, labels=[0,1,2])
    rec = [(cm[i,i]/cm[i,:].sum()) if cm[i,:].sum()>0 else 0 for i in range(3)]
    mba = float(np.mean(rec)); acc = (pred==y).mean(); mf1 = f1_score(y, pred, average="macro")
    return mba, mf1, acc, cm

best = None
for b0 in np.linspace(0, 2.0, 41):    # 对 0=AD 加偏置
  for b1 in np.linspace(0, 2.0, 41):  # 对 1=CN 加偏置
    for T in (0.6,0.7,0.8,0.9,1.0):
      adj = logP / T
      adj[:,0] += b0
      adj[:,1] += b1
      Q = np.exp(adj - adj.max(1, keepdims=True)); Q /= Q.sum(1, keepdims=True)
      key = eval_probs(Q)[:3]
      if (best is None) or (key > best[0]):
        best = (key, (b0, b1, T, eval_probs(Q)[3]))

(b0,b1,T,cm) = best[1]
mba, mf1, acc = best[0]
print(f"[EVAL after bias] b_AD={b0:.2f}  b_CN={b1:.2f}  T={T:.1f}  | Macro-BA={mba:.3f}  Macro-F1={mf1:.3f}  Acc={acc:.3f}  N={len(df)}")
print("Confusion (rows=true, cols=pred, 0/1/2=AD/CN/MCI):")
print(cm)
PY
else
  echo "[ERR] 没有上一步的预测输出，无法做纠偏"; 
fi
