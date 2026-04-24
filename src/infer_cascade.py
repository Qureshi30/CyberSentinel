import argparse
import os

import joblib
import numpy as np
import pandas as pd

from src.cascade_common import (
    artifact_dir,
    load_json,
    load_label_encoder,
    read_dataset,
    stage_features,
    to_binary_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run calibrated two-stage cascade inference")
    parser.add_argument("--model", required=True, choices=["randomforest", "svm", "xgboost"])
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def run_cascade_inference(input_csv, model, output_csv=None):
    out_dir = artifact_dir(model)

    stage1 = joblib.load(os.path.join(out_dir, "stage1_calibrated.pkl"))
    stage2 = joblib.load(os.path.join(out_dir, "stage2_calibrated.pkl"))
    encoder = load_label_encoder(os.path.join(out_dir, "stage2_label_encoder.pkl"))
    cfg = load_json(os.path.join(out_dir, "cascade_config.json"))

    thresholds = cfg["thresholds"]
    t_low = thresholds["stage1_low"]
    t_strong = thresholds["stage1_strong"]
    t_conf = thresholds["stage2_conf"]
    t_margin = thresholds["stage2_margin"]

    raw_df = read_dataset(input_csv)
    X = stage_features(raw_df)

    p_attack = stage1.predict_proba(X)[:, 1]
    p_cat = stage2.predict_proba(X)
    top1 = np.max(p_cat, axis=1)
    top2 = np.partition(p_cat, -2, axis=1)[:, -2]
    margin = top1 - top2
    pred_idx = np.argmax(p_cat, axis=1)
    pred_cat = encoder.inverse_transform(pred_idx)

    final_prediction = []
    decision_path = []

    for i in range(len(X)):
        if p_attack[i] < t_low:
            final_prediction.append("normal")
            decision_path.append("stage1_normal")
            continue

        if p_attack[i] >= t_strong:
            final_prediction.append(pred_cat[i])
            decision_path.append("stage2_attack_strong_stage1")
            continue

        if (top1[i] < t_conf) or (margin[i] < t_margin):
            final_prediction.append("normal")
            decision_path.append("stage2_reverted_normal")
            continue

        final_prediction.append(pred_cat[i])
        decision_path.append("stage2_attack_confident")

    result = raw_df.copy()
    result["stage1_p_attack"] = p_attack
    result["stage1_decision"] = np.where(p_attack >= t_low, "attack_path", "normal")
    result["stage2_pred_category"] = pred_cat
    result["stage2_top1_conf"] = top1
    result["stage2_margin"] = margin
    result["final_prediction"] = final_prediction
    result["decision_path"] = decision_path

    if "label" in raw_df.columns:
        y_true_binary = to_binary_labels(raw_df["label"])
        y_pred_binary = np.array([0 if v == "normal" else 1 for v in final_prediction])
        tp = int(((y_true_binary == 1) & (y_pred_binary == 1)).sum())
        fn = int(((y_true_binary == 1) & (y_pred_binary == 0)).sum())
        fp = int(((y_true_binary == 0) & (y_pred_binary == 1)).sum())
        tn = int(((y_true_binary == 0) & (y_pred_binary == 0)).sum())
        attack_recall = tp / (tp + fn) if (tp + fn) else 0.0
        fpr_normal = fp / (fp + tn) if (fp + tn) else 0.0
        print(
            {
                "attack_recall": round(attack_recall, 6),
                "fpr_normal": round(fpr_normal, 6),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
            }
        )

    output_csv = output_csv
    if not output_csv:
        base, ext = os.path.splitext(input_csv)
        output_csv = f"{base}_cascade_{model}{ext}"

    result.to_csv(output_csv, index=False)
    print(f"Saved predictions: {output_csv}")
    return result


def main():
    args = parse_args()
    run_cascade_inference(
        input_csv=args.input_csv,
        model=args.model,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
