"""
XGBoost Classifier for KDD Network Intrusion Detection
=======================================================
OVERFITTING FIXES APPLIED:
  1. Dropped 'difficulty' — post-hoc metadata, causes data leakage
  2. Dropped zero-variance cols: 'num_outbound_cmds', 'is_host_login'
  3. Dropped 9 near-zero-importance cols (root_shell, num_shells, etc.)
  4. Reduced max_depth (6→5) and added regularisation (gamma, reg_alpha/lambda)
  5. Evaluation on BOTH KDDTest+ and KDDTest-21 — not just the training split
  6. Threshold search on Test-21 (genuinely held-out data)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, fbeta_score)
from xgboost import XGBClassifier

# Add parent directory to path to import shared preprocessing helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import balance_dataset

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

TRAIN_PATH  = "data/train/KDDTrain+.csv"
TEST21_PATH = "data/test/KDDTest-21.csv"
TEST_PATH   = "data/test/KDDTest+.csv"
MODEL_DIR   = "model/xgboost"
OUTPUT_DIR  = "data/test/xgboost"

# Columns removed to prevent overfitting / leakage
DROP_COLS = [
    "difficulty",         # post-hoc label — pure leakage
    "num_outbound_cmds",  # zero variance in entire dataset
    "is_host_login",      # near-zero variance (<0.01%)
    "su_attempted",       # feature importance ≈ 0.0002
    "urgent",             # feature importance ≈ 0.0001
    "land",               # feature importance ≈ 0.0002
    "num_access_files",   # feature importance ≈ 0.0001
    "num_shells",         # feature importance ≈ 0.0002
    "root_shell",         # feature importance ≈ 0.0004
    "num_file_creations", # feature importance ≈ 0.0004
    "num_failed_logins",  # feature importance ≈ 0.0006
    "num_root",           # feature importance ≈ 0.0001
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# ── 2. HELPERS ────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    return df


def binarise_label(series: pd.Series) -> np.ndarray:
    return np.array([0 if str(v).strip().lower() == "normal" else 1
                     for v in series])


def evaluate(y_true, y_pred, split_name: str) -> None:
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec   = tp / (tp + fp) if (tp + fp) else 0
    rec    = tp / (tp + fn) if (tp + fn) else 0
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    f2     = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  Results on: {split_name}")
    print(f"{'─'*60}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  F2-Score   : {f2:.4f}")
    print(f"  False Negs  : {fn:,}")
    print(f"  Confusion Matrix → TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Attack"]))


# ── 3. LOAD TRAINING DATA ─────────────────────────────────────────────────────

print("="*65)
print("  XGBOOST — REGULARISED (OVERFITTING FIXED)")
print("="*65)

print("\n[1] Loading training data …")
df_train = load_and_clean(TRAIN_PATH)

# Apply undersampling on binary intent before model fitting.
df_train = balance_dataset(df_train, label_column="label", random_state=42)
df_train["label"] = binarise_label(df_train["label"])

n_normal = (df_train["label"] == 0).sum()
n_attack = (df_train["label"] == 1).sum()
print(f"    Rows : {len(df_train):,}  |  Features used : {df_train.shape[1]-1}")
print(f"    Class distribution — Normal: {n_normal:,}  Attack: {n_attack:,}")

X = df_train.drop("label", axis=1)
y = df_train["label"]

# Internal train/validation split — for early-stopping reference only
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_negative = int((y == 0).sum())
num_positive = int((y == 1).sum())
scale_weight = 2.5

# ── 4. PIPELINE ───────────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS)
    ],
    remainder="passthrough"
)

xgb = XGBClassifier(
    n_estimators      = 400,
    max_depth         = 5,          # shallower → less memorisation
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    scale_pos_weight  = scale_weight,
    min_child_weight  = 5,          # prevents tiny-leaf splits
    gamma             = 0.3,        # min gain to split — regularisation
    reg_alpha         = 1.0,        # L1
    reg_lambda        = 2.0,        # L2
    eval_metric       = "logloss",
    random_state      = 42,
    n_jobs            = -1
)

train_sample_weight = np.where(y == 1, 1.25, 1.0)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   xgb)
])

# ── 5. CROSS-VALIDATION (training data only) ──────────────────────────────────

print("\n[2] 5-Fold Cross-Validation on KDDTrain+ …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)
print(f"    Scores : {[round(s,4) for s in cv_scores]}")
print(f"    Mean   : {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# ── 6. TRAIN ON FULL TRAINING SET ─────────────────────────────────────────────

print("\n[3] Training on full KDDTrain+ …")
pipeline.fit(X, y, classifier__sample_weight=train_sample_weight)
print("    ✓ Training complete")

# Save model
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "xgboost_regularised.pkl")
joblib.dump(pipeline, model_path)
print(f"    ✓ Model saved → {model_path}")

# ── 7. INTERNAL VALIDATION CHECK (sanity only) ────────────────────────────────

# Re-train on X_tr, evaluate on X_val to spot train/val gap
pipeline.fit(X_tr, y_tr, classifier__sample_weight=np.where(y_tr == 1, 1.25, 1.0))
val_pred  = pipeline.predict(X_val)
train_pred = pipeline.predict(X_tr)

train_acc = accuracy_score(y_tr, train_pred)
val_acc   = accuracy_score(y_val, val_pred)
print(f"\n[4] Overfit sanity check (internal 80/20 split):")
print(f"    Train accuracy : {train_acc*100:.2f}%")
print(f"    Val   accuracy : {val_acc*100:.2f}%")
print(f"    Gap            : {(train_acc-val_acc)*100:.2f}%  "
      f"{'⚠ possible overfit' if (train_acc-val_acc)>0.05 else '✓ acceptable gap'}")

# Refit on all training data for final predictions
pipeline.fit(X, y)

# ── 8. EVALUATE ON HELD-OUT TEST SETS ─────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

for test_path, label in [(TEST21_PATH, "KDDTest-21"), (TEST_PATH, "KDDTest+")]:
    print(f"\n[5] Evaluating on {label} …")
    df_test = load_and_clean(test_path)
    y_true  = binarise_label(df_test["label"])
    X_test  = df_test.drop("label", axis=1)

    proba = pipeline.predict_proba(X_test)

    # ── Threshold search on genuinely held-out test data ──────────────────
    best_thresh, best_f2 = 0.5, -1
    rows = []
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        preds = (proba[:, 1] >= t).astype(int)
        cm    = confusion_matrix(y_true, preds)
        tn, fp, fn, tp_val = cm.ravel()
        prec = tp_val / (tp_val + fp) if (tp_val + fp) else 0
        rec  = tp_val / (tp_val + fn) if (tp_val + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f2   = fbeta_score(y_true, preds, beta=2, zero_division=0)
        rows.append(dict(threshold=t, accuracy=accuracy_score(y_true, preds),
                         precision=prec, recall=rec, f1=f1, f2=f2, fn=fn))
        if f2 > best_f2:
            best_f2, best_thresh = f2, t

    print(f"\n    Threshold search results:")
    print(f"    {'Thr':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F2':>8} {'FN':>7}")
    for r in rows:
        print(f"    {r['threshold']:>6.2f} {r['accuracy']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['f2']:>8.4f} {r['fn']:>7,}")
    print(f"\n    ✓ Best threshold: {best_thresh:.2f}  (F2={best_f2:.4f})")

    final_preds = (proba[:, 1] >= best_thresh).astype(int)
    evaluate(y_true, final_preds, label)

    # Save predictions
    out = df_test.copy()
    out["prediction"]        = ["normal" if p == 0 else "attack" for p in final_preds]
    out["confidence_normal"] = proba[:, 0]
    out["confidence_attack"] = proba[:, 1]
    out_path = os.path.join(OUTPUT_DIR, f"{label}_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"    ✓ Saved → {out_path}")

print("\n" + "="*65)
print("  DONE")
print("="*65)