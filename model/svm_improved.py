"""
Improved SVM Model for Intrusion Detection
Key improvements over baseline:
1. Uses FULL KDDTrain+ dataset (125K rows) instead of 20%
2. Feature scaling on all numeric columns (critical for SVM)
3. Hyperparameter tuning (C, gamma) via GridSearchCV
4. Evaluates on both internal split AND KDDTest-21 external test set
5. Detailed per-class analysis
"""

import pandas as pd
import numpy as np
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.svm import SVC


def load_and_prepare(csv_path):
    """Load CSV and prepare features/labels"""
    df = pd.read_csv(csv_path)
    
    if "difficulty" in df.columns:
        df = df.drop("difficulty", axis=1)
    
    # Binary classification: normal=0, attack=1
    df["label"] = df["label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
    
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y


def build_svm_pipeline(C=10, gamma='scale'):
    """Build pipeline with full preprocessing + SVM"""
    categorical_cols = ["protocol_type", "service", "flag"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"  # numeric columns pass through
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),  # Scale ALL features (critical for SVM)
        ("classifier", SVC(
            kernel="rbf",
            C=C,
            gamma=gamma,
            class_weight="balanced",
            random_state=42
        ))
    ])
    return pipeline


def main():
    print("=" * 60)
    print("IMPROVED SVM - INTRUSION DETECTION")
    print("=" * 60)
    
    # ── Step 1: Load FULL training data ──
    print("\n[1/5] Loading FULL training dataset...")
    X_train_full, y_train_full = load_and_prepare("data/train/KDDTrain+.csv")
    print(f"   ✓ Training data: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"   ✓ Attacks: {sum(y_train_full)}, Normal: {len(y_train_full) - sum(y_train_full)}")
    
    # ── Step 2: Load external test set ──
    print("\n[2/5] Loading KDDTest-21 test dataset...")
    X_test_ext, y_test_ext = load_and_prepare("data/test/KDDTest-21.csv")
    print(f"   ✓ Test data: {X_test_ext.shape[0]} samples")
    print(f"   ✓ Attacks: {sum(y_test_ext)}, Normal: {len(y_test_ext) - sum(y_test_ext)}")
    
    # ── Step 3: Internal validation split ──
    print("\n[3/5] Creating internal validation split (80/20)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"   ✓ Train: {len(X_train)}, Validation: {len(X_val)}")
    
    # ── Step 4: Hyperparameter tuning ──
    print("\n[4/5] Hyperparameter tuning via GridSearchCV...")
    print("   Testing C=[1, 10, 100] and gamma=['scale', 0.01, 0.001]")
    print("   This may take several minutes...\n")
    
    param_grid = {
        'classifier__C': [1, 10, 100],
        'classifier__gamma': ['scale', 0.01, 0.001],
    }
    
    base_pipeline = build_svm_pipeline()
    
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=3,           # 3-fold cross-validation
        scoring='f1',   # Optimize for F1 (balances precision & recall)
        n_jobs=-1,      # Use all CPU cores
        verbose=1
    )
    
    start = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start
    
    print(f"\n   ✓ Grid search completed in {elapsed:.1f}s")
    print(f"   ✓ Best parameters: {grid_search.best_params_}")
    print(f"   ✓ Best CV F1 score: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # ── Step 5: Evaluate ──
    print("\n[5/5] Evaluating best model...")
    
    # Internal validation
    y_val_pred = best_model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\n   ── Internal Validation (20% holdout) ──")
    print(f"   Accuracy: {val_acc * 100:.2f}%")
    print(classification_report(y_val, y_val_pred, target_names=["Normal", "Attack"]))
    
    # External test (KDDTest-21)
    y_ext_pred = best_model.predict(X_test_ext)
    ext_acc = accuracy_score(y_test_ext, y_ext_pred)
    print(f"   ── External Test (KDDTest-21) ──")
    print(f"   Accuracy: {ext_acc * 100:.2f}%")
    print(classification_report(y_test_ext, y_ext_pred, target_names=["Normal", "Attack"]))
    
    cm = confusion_matrix(y_test_ext, y_ext_pred)
    print(f"   Confusion Matrix:")
    print(f"                   Predicted Normal  Predicted Attack")
    print(f"   Actual Normal   {cm[0][0]:>15}  {cm[0][1]:>16}")
    print(f"   Actual Attack   {cm[1][0]:>15}  {cm[1][1]:>16}")
    
    # ── Retrain on FULL data with best params, then save ──
    print("\n   Retraining on FULL training data with best parameters...")
    best_C = grid_search.best_params_['classifier__C']
    best_gamma = grid_search.best_params_['classifier__gamma']
    
    final_pipeline = build_svm_pipeline(C=best_C, gamma=best_gamma)
    final_pipeline.fit(X_train_full, y_train_full)
    
    # Final evaluation on external test
    y_final_pred = final_pipeline.predict(X_test_ext)
    final_acc = accuracy_score(y_test_ext, y_final_pred)
    print(f"   ✓ Final model (full data) accuracy on KDDTest-21: {final_acc * 100:.2f}%")
    print(classification_report(y_test_ext, y_final_pred, target_names=["Normal", "Attack"]))
    
    # Save
    joblib.dump(final_pipeline, "model/svm.pkl")
    print("   ✓ Model saved to model/svm.pkl")
    
    # ── Summary ──
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Internal Validation Accuracy:  {val_acc * 100:.2f}%")
    print(f"  KDDTest-21 Accuracy (tuned):   {ext_acc * 100:.2f}%")
    print(f"  KDDTest-21 Accuracy (full):    {final_acc * 100:.2f}%")
    print(f"  Best C: {best_C}, Best gamma: {best_gamma}")
    print("=" * 60)


if __name__ == "__main__":
    main()
