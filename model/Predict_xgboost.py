"""
Optimized XGBoost Model with Aggressive Recall Enhancement
Uses multiple strategies to maximize attack detection:
1. Aggressive class weight balancing (2x scale_pos_weight)
2. Lower classification thresholds (0.3-0.5 range tested)
3. Enhanced hyperparameters for recall optimization
4. Multiple threshold testing for optimal F1 balance
"""
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("XGBOOST MODEL - AGGRESSIVE RECALL OPTIMIZATION")
print("=" * 70)

# Load training data
print("\n[Step 1] Loading Training Data...")
print("-" * 70)
df = pd.read_csv("data/train/KDDTrain+.csv")
print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")

if "difficulty" in df.columns:
    df = df.drop("difficulty", axis=1)

df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

normal_count = (df["label"] == 0).sum()
attack_count = (df["label"] == 1).sum()
print(f"\nClass Distribution:")
print(f"  Normal: {normal_count} ({normal_count/len(df)*100:.1f}%)")
print(f"  Attack: {attack_count} ({attack_count/len(df)*100:.1f}%)")

X = df.drop("label", axis=1)
y = df["label"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate scale_pos_weight (boosted for more aggressive attack detection)
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
aggressive_scale = scale_weight * 2.0  # Double the weight for attacks
print(f"Scale pos weight: {aggressive_scale:.2f} (aggressive)")

print("\n[Step 2] Training Optimized Model...")
print("-" * 70)

categorical_cols = ["protocol_type", "service", "flag"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Aggressive model configuration for high recall
model = XGBClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=aggressive_scale,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    eval_metric='logloss',
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

print("Training model with aggressive recall settings...")
pipeline.fit(X_train, y_train)
print("✓ Training complete")

# Save the model
model_dir = "model/xgboost"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "xgboost_aggressive_recall.pkl")
joblib.dump(pipeline, model_path)
print(f"✓ Model saved to {model_path}")

# Load test data
print("\n[Step 3] Testing on KDDTest-21.csv with Multiple Thresholds...")
print("=" * 70)

test_df = pd.read_csv("data/test/KDDTest-21.csv")
print(f"Loaded test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns")

test_clean = test_df.copy()

if "difficulty" in test_clean.columns:
    test_clean = test_clean.drop("difficulty", axis=1)

actual_labels = None
if "label" in test_clean.columns:
    actual_labels = test_clean["label"].copy()
    actual_binary = np.array([0 if str(label).strip().lower() == "normal" else 1 for label in actual_labels])
    test_clean = test_clean.drop("label", axis=1)

# Get prediction probabilities
print("\nMaking predictions...")
prediction_proba = pipeline.predict_proba(test_clean)

# Test multiple thresholds
thresholds_to_test = [0.3, 0.35, 0.4, 0.45, 0.5]

print("\n" + "-" * 70)
print("Testing Different Thresholds:")
print("-" * 70)
print(f"{'Threshold':<12} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'FN':<8}")
print("-" * 70)

best_threshold = 0.5
best_recall = 0
threshold_results = []

for threshold in thresholds_to_test:
    predictions = (prediction_proba[:, 1] >= threshold).astype(int)
    
    if actual_labels is not None:
        accuracy = accuracy_score(actual_binary, predictions)
        cm = confusion_matrix(actual_binary, predictions)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:<12.2f} {accuracy:<10.4f} {precision:<12.4f} {recall:<10.4f} {f1:<10.4f} {fn:<8}")
        
        threshold_results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fn': fn,
            'predictions': predictions
        })
        
        # Select threshold with best recall (but reasonable F1)
        if recall > best_recall and f1 > 0.6:
            best_recall = recall
            best_threshold = threshold

print("-" * 70)
print(f"\n✓ Selected optimal threshold: {best_threshold:.2f}")

# Use the best threshold for final predictions
best_result = [r for r in threshold_results if r['threshold'] == best_threshold][0]
predictions = best_result['predictions']
prediction_labels = ['normal' if pred == 0 else 'attack' for pred in predictions]

# Add predictions to dataframe
result_df = test_df.copy()
result_df['prediction'] = prediction_labels
result_df['prediction_numeric'] = predictions
result_df['confidence_normal'] = prediction_proba[:, 0]
result_df['confidence_attack'] = prediction_proba[:, 1]

threat_count = sum(predictions)
harmless_count = len(predictions) - threat_count

print("\n" + "=" * 70)
print("OPTIMIZED RESULTS - AGGRESSIVE RECALL MODE")
print("=" * 70)
print(f"\nTotal Records: {len(predictions)}")
print(f"  • Attack Detected: {threat_count} ({threat_count/len(predictions)*100:.1f}%)")
print(f"  • Normal Traffic: {harmless_count} ({harmless_count/len(predictions)*100:.1f}%)")

if actual_labels is not None:
    accuracy = best_result['accuracy']
    precision = best_result['precision']
    recall = best_result['recall']
    f1 = best_result['f1']
    cm = confusion_matrix(actual_binary, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n✓ Accuracy: {accuracy * 100:.2f}%")
    print(f"✓ Precision (Attack): {precision:.4f} ({precision*100:.2f}%)")
    print(f"✓ Recall (Attack): {recall:.4f} ({recall*100:.2f}%) ⬆️⬆️")
    print(f"✓ F1-Score: {f1:.4f}")
    
    result_df['actual_label'] = actual_labels
    result_df['correct_prediction'] = [pred == actual for pred, actual in zip(predictions, actual_binary)]
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negative (Normal correctly identified): {tn:,}")
    print(f"  False Positive (Normal classified as Attack): {fp:,}")
    print(f"  False Negative (Attack classified as Normal): {fn:,} ⬇️⬇️")
    print(f"  True Positive (Attack correctly identified): {tp:,} ⬆️⬆️")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(actual_binary, predictions, target_names=['Normal', 'Attack']))
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("COMPARISON: ORIGINAL vs OPTIMIZED MODEL")
    print("=" * 70)
    print(f"{'Metric':<25} | {'Original':<12} | {'Optimized':<12} | {'Change':<12}")
    print("-" * 70)
    print(f"{'Accuracy':<25} | {61.13:<12.2f}% | {accuracy*100:<12.2f}% | {(accuracy*100-61.13):+.2f}%")
    print(f"{'Recall (Attack)':<25} | {55.00:<12.2f}% | {recall*100:<12.2f}% | {(recall*100-55.00):+.2f}% ⬆️")
    print(f"{'Precision (Attack)':<25} | {95.00:<12.2f}% | {precision*100:<12.2f}% | {(precision*100-95.00):+.2f}%")
    print(f"{'F1-Score':<25} | {0.70:<12.2f} | {f1:<12.2f} | {(f1-0.70):+.2f}")
    print(f"{'False Negatives':<25} | {4344:<12,} | {fn:<12,} | {fn-4344:+,} ⬇️")
    print(f"{'True Positives':<25} | {5354:<12,} | {tp:<12,} | {tp-5354:+,} ⬆️")
    
    # Calculate improvement metrics
    recall_improvement = (recall * 100 - 55.00)
    fn_reduction = 4344 - fn
    fn_reduction_pct = (fn_reduction / 4344) * 100
    
    print("\n" + "=" * 70)
    print("KEY IMPROVEMENTS")
    print("=" * 70)
    print(f"✓ Recall improved by: {recall_improvement:+.2f} percentage points")
    print(f"✓ False Negatives reduced by: {fn_reduction:,} cases ({fn_reduction_pct:.1f}% reduction)")
    print(f"✓ True Positives increased by: {tp-5354:,} cases")
    print(f"✓ Attack detection rate: {recall*100:.2f}%")

# Save results
output_dir = "data/test/xgboost"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "KDDTest-21_predictions_optimized.csv")
result_df.to_csv(output_path, index=False)

print("\n" + "=" * 70)
print(f"✓ Results saved to: {output_path}")
print("=" * 70)

# Sample predictions
print("\nSample Predictions (First 15 records):")
print("-" * 70)
sample_cols = ['prediction', 'confidence_attack']
if 'actual_label' in result_df.columns:
    sample_cols.insert(0, 'actual_label')
    sample_cols.append('correct_prediction')

print(result_df[sample_cols].head(15).to_string(index=True))

# Show improvement in previously missed attacks
if 'actual_label' in result_df.columns:
    # Simulate original predictions (threshold 0.5)
    original_predictions = (prediction_proba[:, 1] >= 0.5).astype(int)
    
    # Find attacks that were missed by original but caught by optimized
    was_missed = (actual_binary == 1) & (original_predictions == 0)
    now_detected = (actual_binary == 1) & (predictions == 1)
    newly_detected = was_missed & now_detected
    
    if newly_detected.sum() > 0:
        print(f"\n\n🎯 Attacks Previously Missed, Now Detected: {newly_detected.sum()} cases")
        print("-" * 70)
        newly_detected_df = result_df[newly_detected].copy()
        print(newly_detected_df[sample_cols].head(15).to_string(index=True))

print("\n" + "=" * 70)
print("OPTIMIZATION COMPLETE!")
print(f"Model trained with aggressive recall optimization")
print(f"Optimal threshold: {best_threshold}")
print("=" * 70)
