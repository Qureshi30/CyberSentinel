import argparse
import os

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

from src.cascade_common import (
    artifact_dir,
    load_label_encoder,
    read_dataset,
    save_json,
    split_train_val,
    stage_features,
    to_attack_category,
    to_binary_labels,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate Stage 1 / Stage 2 models")
    parser.add_argument("--model", required=True, choices=["randomforest", "svm", "xgboost"])
    parser.add_argument("--stage", default="both", choices=["stage1", "stage2", "both"])
    parser.add_argument("--method", default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--train-path", default=os.path.join("data", "train", "KDDTrain+.csv"))
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def calibrate_stage1(out_dir, X_val, y_val, method):
    base_model = joblib.load(os.path.join(out_dir, "stage1_model.pkl"))
    calibrated = _build_prefit_calibrator(base_model, method)
    calibrated.fit(X_val, y_val)
    joblib.dump(calibrated, os.path.join(out_dir, "stage1_calibrated.pkl"))

    proba = calibrated.predict_proba(X_val)[:, 1]
    brier = brier_score_loss(y_val, proba)
    return {"brier_score": float(brier), "method": method}


def calibrate_stage2(out_dir, X_val_attack, y_val_attack, method):
    base_model = joblib.load(os.path.join(out_dir, "stage2_model.pkl"))
    calibrated = _build_prefit_calibrator(base_model, method)
    calibrated.fit(X_val_attack, y_val_attack)
    joblib.dump(calibrated, os.path.join(out_dir, "stage2_calibrated.pkl"))

    proba = calibrated.predict_proba(X_val_attack)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_val_attack)), y_val_attack] = 1.0
    brier_multi = np.mean(np.sum((proba - one_hot) ** 2, axis=1))
    return {"multiclass_brier": float(brier_multi), "method": method}


def _build_prefit_calibrator(base_model, method):
    """
    Build a calibrator compatible with both newer and older scikit-learn APIs.
    """
    try:
        from sklearn.frozen import FrozenEstimator

        return CalibratedClassifierCV(
            estimator=FrozenEstimator(base_model),
            method=method,
            cv=None,
        )
    except Exception:
        return CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")


def main():
    args = parse_args()
    out_dir = artifact_dir(args.model)

    df = read_dataset(args.train_path)
    _, val_df = split_train_val(df, val_size=args.val_size, random_state=args.random_state)

    calibration_metrics = {}

    if args.stage in ["stage1", "both"]:
        X_val = stage_features(val_df)
        y_val = to_binary_labels(val_df["label"])
        calibration_metrics["stage1"] = calibrate_stage1(out_dir, X_val, y_val, args.method)

    if args.stage in ["stage2", "both"]:
        y_val_bin = to_binary_labels(val_df["label"])
        val_attack_df = val_df[y_val_bin == 1].copy()
        val_attack_df["attack_category"] = to_attack_category(val_attack_df["label"])
        X_val_attack = stage_features(val_attack_df)

        encoder = load_label_encoder(os.path.join(out_dir, "stage2_label_encoder.pkl"))
        y_val_attack = encoder.transform(val_attack_df["attack_category"])

        calibration_metrics["stage2"] = calibrate_stage2(
            out_dir,
            X_val_attack,
            y_val_attack,
            args.method,
        )

    save_json(os.path.join(out_dir, "calibration_metrics.json"), calibration_metrics)
    print("Calibration complete")
    print(calibration_metrics)


if __name__ == "__main__":
    main()
