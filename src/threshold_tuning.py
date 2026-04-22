import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import f1_score

from src.cascade_common import (
    artifact_dir,
    load_label_encoder,
    read_dataset,
    save_json,
    split_train_val,
    stage_features,
    summarize_binary,
    to_binary_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune cascade thresholds on validation data")
    parser.add_argument("--model", required=True, choices=["randomforest", "svm", "xgboost"])
    parser.add_argument("--train-path", default=os.path.join("data", "train", "KDDTrain+.csv"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-attack-recall", type=float, default=0.90)
    return parser.parse_args()


def select_best(rows, min_attack_recall):
    valid = [r for r in rows if r["attack_recall"] >= min_attack_recall]
    if not valid:
        return max(rows, key=lambda r: (r["attack_recall"], -r["fpr_normal"]))
    return min(valid, key=lambda r: (r["fpr_normal"], -r["attack_recall"], -r["binary_f1"]))


def main():
    args = parse_args()
    out_dir = artifact_dir(args.model)

    stage1 = joblib.load(os.path.join(out_dir, "stage1_calibrated.pkl"))
    stage2 = joblib.load(os.path.join(out_dir, "stage2_calibrated.pkl"))
    label_encoder = load_label_encoder(os.path.join(out_dir, "stage2_label_encoder.pkl"))

    df = read_dataset(args.train_path)
    _, val_df = split_train_val(df, val_size=args.val_size, random_state=args.random_state)

    X_val = stage_features(val_df)
    y_val_binary = to_binary_labels(val_df["label"])

    p_attack = stage1.predict_proba(X_val)[:, 1]
    p_cat = stage2.predict_proba(X_val)
    top1 = np.max(p_cat, axis=1)
    top2 = np.partition(p_cat, -2, axis=1)[:, -2]
    margin = top1 - top2
    stage2_pred_idx = np.argmax(p_cat, axis=1)
    stage2_pred_label = label_encoder.inverse_transform(stage2_pred_idx)

    t_stage1_low_grid = np.arange(0.20, 0.76, 0.05)
    t_stage1_strong_grid = np.arange(0.75, 0.96, 0.05)
    t_stage2_conf_grid = np.arange(0.30, 0.91, 0.05)
    t_stage2_margin_grid = np.arange(0.05, 0.46, 0.05)

    rows = []

    for t_low in t_stage1_low_grid:
        for t_strong in t_stage1_strong_grid:
            if t_strong <= t_low:
                continue
            for t_conf in t_stage2_conf_grid:
                for t_margin in t_stage2_margin_grid:
                    final_binary = np.zeros(len(X_val), dtype=int)

                    attack_gate = p_attack >= t_low
                    strong_attack = p_attack >= t_strong
                    weak_mod_attack = attack_gate & (~strong_attack)

                    # Strong Stage 1 attack signals stay attack
                    final_binary[strong_attack] = 1

                    # Weak/moderate stage1 attacks require stage2 confidence and margin
                    keep_attack = weak_mod_attack & (top1 >= t_conf) & (margin >= t_margin)
                    final_binary[keep_attack] = 1

                    m = summarize_binary(y_val_binary, final_binary)
                    m["binary_f1"] = float(f1_score(y_val_binary, final_binary, zero_division=0))
                    m["stage1_low"] = float(round(t_low, 4))
                    m["stage1_strong"] = float(round(t_strong, 4))
                    m["stage2_conf"] = float(round(t_conf, 4))
                    m["stage2_margin"] = float(round(t_margin, 4))
                    rows.append(m)

    best = select_best(rows, min_attack_recall=args.min_attack_recall)

    config = {
        "model": args.model,
        "thresholds": {
            "stage1_low": best["stage1_low"],
            "stage1_strong": best["stage1_strong"],
            "stage2_conf": best["stage2_conf"],
            "stage2_margin": best["stage2_margin"],
        },
        "validation_metrics": {
            "attack_recall": best["attack_recall"],
            "fpr_normal": best["fpr_normal"],
            "binary_f1": best["binary_f1"],
            "accuracy": best["accuracy"],
        },
    }

    save_json(os.path.join(out_dir, "cascade_config.json"), config)

    # Save a compact leaderboard for inspection
    leaderboard = sorted(rows, key=lambda r: (r["fpr_normal"], -r["attack_recall"]))[:20]
    save_json(os.path.join(out_dir, "threshold_leaderboard_top20.json"), {"rows": leaderboard})

    print("Threshold tuning complete")
    print(config)


if __name__ == "__main__":
    main()
