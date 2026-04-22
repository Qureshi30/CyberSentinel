import argparse
import os

import joblib
from sklearn.preprocessing import LabelEncoder

from src.cascade_common import (
    artifact_dir,
    build_stage2_pipeline,
    fit_with_sample_weights,
    read_dataset,
    save_json,
    save_label_encoder,
    split_train_val,
    stage_features,
    summarize_multiclass,
    to_attack_category,
    to_binary_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 2 attack category model")
    parser.add_argument("--model", required=True, choices=["randomforest", "svm", "xgboost"])
    parser.add_argument("--train-path", default=os.path.join("data", "train", "KDDTrain+.csv"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    df = read_dataset(args.train_path)
    train_df, val_df = split_train_val(df, val_size=args.val_size, random_state=args.random_state)

    y_train_bin = to_binary_labels(train_df["label"])
    y_val_bin = to_binary_labels(val_df["label"])

    train_attack_df = train_df[y_train_bin == 1].copy()
    val_attack_df = val_df[y_val_bin == 1].copy()

    if train_attack_df.empty or val_attack_df.empty:
        raise ValueError("Attack-only train/val subsets are empty. Check your data split.")

    train_attack_df["attack_category"] = to_attack_category(train_attack_df["label"])
    val_attack_df["attack_category"] = to_attack_category(val_attack_df["label"])

    X_train = stage_features(train_attack_df)
    X_val = stage_features(val_attack_df)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_attack_df["attack_category"])
    y_val = label_encoder.transform(val_attack_df["attack_category"])

    pipeline = build_stage2_pipeline(args.model, X_train)
    pipeline = fit_with_sample_weights(pipeline, X_train, y_train)

    val_pred = pipeline.predict(X_val)
    metrics = summarize_multiclass(y_val, val_pred, class_names=label_encoder.classes_)

    out_dir = artifact_dir(args.model)
    joblib.dump(pipeline, os.path.join(out_dir, "stage2_model.pkl"))
    save_label_encoder(os.path.join(out_dir, "stage2_label_encoder.pkl"), label_encoder)
    save_json(
        os.path.join(out_dir, "stage2_metrics.json"),
        {
            "model": args.model,
            "train_path": args.train_path,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "classes": label_encoder.classes_.tolist(),
            "validation": metrics,
        },
    )

    print("Stage 2 training complete")
    print(f"Saved model: {os.path.join(out_dir, 'stage2_model.pkl')}")
    print(f"Saved label encoder: {os.path.join(out_dir, 'stage2_label_encoder.pkl')}")
    print(f"Validation macro F1: {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
