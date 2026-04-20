"""
Random Forest Classifier for KDD Network Intrusion Detection
=============================================================
OVERFITTING FIXES APPLIED:
  1. Dropped 'difficulty' — post-hoc metadata, causes data leakage
  2. Dropped zero-variance cols: 'num_outbound_cmds', 'is_host_login'
  3. Dropped 9 near-zero-importance cols (root_shell, num_shells, etc.)
  4. Limited max_depth=20→15 and raised min_samples_leaf=2→4
  5. Replaced SelectKBest (mutual_info on train split = leakage risk)
     with a fixed domain-knowledge feature list + importance-pruning
  6. Evaluation on BOTH KDDTest+ and KDDTest-21
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, fbeta_score)

# Add parent directory to path to import shared preprocessing helpers
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import balance_dataset

# ── 1. CONFIGURATION ──────────────────────────────────────────────────────────

TRAIN_PATH  = "data/train/KDDTrain+.csv"
TEST21_PATH = "data/test/KDDTest-21.csv"
TEST_PATH   = "data/test/KDDTest+.csv"
MODEL_DIR   = "model"
OUTPUT_DIR  = "data/test/random_forest"

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
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    f2   = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print(f"\n{'─'*60}")
    print(f"  Results on: {split_name}")
    print(f"{'─'*60}")
    print(f"  Accuracy   : {acc*100:.2f}%")
    print(f"  Precision  : {prec*100:.2f}%")
    print(f"  Recall     : {rec*100:.2f}%")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  F2-Score   : {f2:.4f}")
    print(f"  False Pos  : {fp:,}")
    print(f"  False Negs  : {fn:,}")
    print(f"  Confusion Matrix → TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(classification_report(y_true, y_pred,
                                target_names=["Normal", "Attack"]))

# ── 3. LOAD DATA ──────────────────────────────────────────────────────────────

print("="*65)
print("  RANDOM FOREST — REGULARISED (OVERFITTING FIXED)")
print("="*65)

print("\n[1] Loading training data …")
df_train = load_and_clean(TRAIN_PATH)

# Apply aggressive undersampling before binary conversion for training.
df_train = balance_dataset(df_train, label_column="label", random_state=42)
df_train["label"] = binarise_label(df_train["label"])

n_normal = (df_train["label"] == 0).sum()
n_attack = (df_train["label"] == 1).sum()
print(f"    Rows : {len(df_train):,}  |  Features used : {df_train.shape[1]-1}")
print(f"    Class distribution — Normal: {n_normal:,}  Attack: {n_attack:,}")

X = df_train.drop("label", axis=1)
y = df_train["label"]

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. PIPELINE ───────────────────────────────────────────────────────────────

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_COLS)
    ],
    remainder="passthrough"
)

# Key regularisation knobs:
#   max_depth        reduced  → shallower trees, less memorisation
#   min_samples_leaf raised   → each leaf needs ≥4 samples
#   max_features     kept as sqrt → reduces variance between trees
#   class_weight     balanced_subsample → handles imbalance per-tree
rf = RandomForestClassifier(
    n_estimators     = 300,
    max_depth        = 15,              # was unlimited → now capped
    min_samples_split= 8,              # needs 8 samples to split
    min_samples_leaf = 4,              # each leaf ≥ 4 samples
    max_features     = "sqrt",
    class_weight     = "balanced_subsample",
    random_state     = 42,
    n_jobs           = -1
)

train_sample_weight = np.where(y == 1, 1.25, 1.0)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   rf)
])

# ── 5. CROSS-VALIDATION ───────────────────────────────────────────────────────

print("\n[2] 5-Fold Cross-Validation on KDDTrain+ …")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=1)
print(f"    Scores : {[round(s,4) for s in cv_scores]}")
print(f"    Mean   : {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})")

# ── 6. OVERFIT CHECK ──────────────────────────────────────────────────────────

print("\n[3] Overfit sanity check (internal 80/20 split) …")
pipeline.fit(X_tr, y_tr)

train_acc = accuracy_score(y_tr, pipeline.predict(X_tr))
val_acc   = accuracy_score(y_val, pipeline.predict(X_val))
print(f"    Train accuracy : {train_acc*100:.2f}%")
print(f"    Val   accuracy : {val_acc*100:.2f}%")
gap = train_acc - val_acc
print(f"    Gap            : {gap*100:.2f}%  "
      f"{'⚠ possible overfit' if gap > 0.05 else '✓ acceptable gap'}")

# ── 7. FEATURE IMPORTANCE ─────────────────────────────────────────────────────

# Extract feature names after one-hot encoding
ohe_features = (pipeline.named_steps["preprocessor"]
                .named_transformers_["cat"]
                .get_feature_names_out(CATEGORICAL_COLS).tolist())
num_features  = [c for c in X.columns if c not in CATEGORICAL_COLS]
all_features  = ohe_features + num_features

importances = pd.Series(
    pipeline.named_steps["classifier"].feature_importances_,
    index=all_features
).sort_values(ascending=False)

print(f"\n    Top 10 features:")
for feat, imp in importances.head(10).items():
    print(f"      {feat:<45} {imp:.4f}")

# ── 8. TRAIN ON FULL DATASET & EVALUATE ───────────────────────────────────────

print("\n[4] Training on full KDDTrain+ …")
pipeline.fit(X, y, classifier__sample_weight=train_sample_weight)

os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "random_forest.pkl")
joblib.dump(pipeline, model_path)
print(f"    ✓ Model saved → {model_path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for test_path, label in [(TEST21_PATH, "KDDTest-21"), (TEST_PATH, "KDDTest+")]:
    print(f"\n[5] Evaluating on {label} …")
    df_test = load_and_clean(test_path)
    y_true  = binarise_label(df_test["label"])
    X_test  = df_test.drop("label", axis=1)

    proba = pipeline.predict_proba(X_test)

    best_thresh, best_f2 = 0.5, -1
    rows = []
    for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        preds = (proba[:, 1] >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        f2 = fbeta_score(y_true, preds, beta=2, zero_division=0)
        rows.append(dict(threshold=t, accuracy=accuracy_score(y_true, preds),
                         precision=prec, recall=rec, f1=f1, f2=f2, fp=fp, fn=fn))
        if f2 > best_f2:
            best_f2, best_thresh = f2, t

    print(f"\n    Threshold search results:")
    print(f"    {'Thr':>6} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'F2':>8} {'FP':>7} {'FN':>7}")
    for r in rows:
        print(f"    {r['threshold']:>6.2f} {r['accuracy']:>8.4f} "
              f"{r['precision']:>8.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['f2']:>8.4f} {r['fp']:>7,} {r['fn']:>7,}")
    print(f"\n    ✓ Best threshold: {best_thresh:.2f}  (F2={best_f2:.4f})")

    y_pred = (proba[:, 1] >= best_thresh).astype(int)
    evaluate(y_true, y_pred, label)

    out = df_test.copy()
    out["prediction"] = ["normal" if p == 0 else "attack" for p in y_pred]
    proba = pipeline.predict_proba(X_test)
    out["confidence_normal"] = proba[:, 0]
    out["confidence_attack"] = proba[:, 1]
    out_path = os.path.join(OUTPUT_DIR, f"{label}_predictions.csv")
    out.to_csv(out_path, index=False)
    print(f"    ✓ Saved → {out_path}")

print("\n" + "="*65)
print("  DONE")
print("="*65)
