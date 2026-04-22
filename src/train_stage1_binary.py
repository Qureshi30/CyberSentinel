import argparse
import os

import joblib

from src.cascade_common import (
    artifact_dir,
    build_stage1_pipeline,
    read_dataset,
    save_json,
    split_train_val,
    stage_features,
    summarize_binary,
    to_binary_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 binary model (normal vs attack)")
    parser.add_argument("--model", required=True, choices=["randomforest", "svm", "xgboost"])
    parser.add_argument("--train-path", default=os.path.join("data", "train", "KDDTrain+.csv"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    df = read_dataset(args.train_path)
    train_df, val_df = split_train_val(df, val_size=args.val_size, random_state=args.random_state)

    X_train = stage_features(train_df)
    y_train = to_binary_labels(train_df["label"])
    X_val = stage_features(val_df)
    y_val = to_binary_labels(val_df["label"])

    pipeline = build_stage1_pipeline(args.model, X_train, y_train)
    pipeline.fit(X_train, y_train)

    val_pred = pipeline.predict(X_val)
    metrics = summarize_binary(y_val, val_pred)

    out_dir = artifact_dir(args.model)
    joblib.dump(pipeline, os.path.join(out_dir, "stage1_model.pkl"))
    train_df.to_csv(os.path.join(out_dir, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "split_val.csv"), index=False)
    save_json(
        os.path.join(out_dir, "stage1_metrics.json"),
        {
            "model": args.model,
            "train_path": args.train_path,
            "val_size": args.val_size,
            "random_state": args.random_state,
            "validation": metrics,
        },
    )

    print("Stage 1 training complete")
    print(f"Saved model: {os.path.join(out_dir, 'stage1_model.pkl')}")
    print(f"Validation metrics: {metrics}")


if __name__ == "__main__":
    main()
