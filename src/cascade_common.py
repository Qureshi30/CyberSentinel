import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.preprocess import clean_data, load_csv

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

DROP_COLS = [
    "difficulty",
    "num_outbound_cmds",
    "is_host_login",
    "su_attempted",
    "urgent",
    "land",
    "num_access_files",
    "num_shells",
    "root_shell",
    "num_file_creations",
    "num_failed_logins",
    "num_root",
]

ATTACK_TO_CATEGORY = {
    # DoS
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "mailbomb": "DoS",
    "processtable": "DoS",
    "udpstorm": "DoS",
    "worm": "DoS",
    # Probe
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    # R2L
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    # U2R
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def artifact_dir(model_name: str) -> str:
    out = os.path.join("model", "cascade", model_name.lower())
    ensure_dir(out)
    return out


def read_dataset(path: str) -> pd.DataFrame:
    df = load_csv(path)
    if df is None:
        raise ValueError(f"Unable to load dataset: {path}")
    df = clean_data(df.copy())
    drop_now = [c for c in DROP_COLS if c in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)
    if "label" not in df.columns:
        raise ValueError("Dataset must include a 'label' column")
    return df


def to_binary_labels(label_series: pd.Series) -> np.ndarray:
    return np.array([0 if str(v).strip().lower() == "normal" else 1 for v in label_series])


def to_attack_category(label_series: pd.Series) -> pd.Series:
    categories = []
    for value in label_series:
        label = str(value).strip().lower()
        if label == "normal":
            categories.append("NORMAL")
        else:
            categories.append(ATTACK_TO_CATEGORY.get(label, "R2L"))
    return pd.Series(categories, index=label_series.index)


def stage_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["label", "attack_category"], errors="ignore")


def build_preprocessor(df_features: pd.DataFrame) -> ColumnTransformer:
    active_cats = [c for c in CATEGORICAL_COLS if c in df_features.columns]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), active_cats),
        ],
        remainder="passthrough",
    )


def compute_scale_pos_weight(y_binary: np.ndarray) -> float:
    n_neg = int((y_binary == 0).sum())
    n_pos = int((y_binary == 1).sum())
    if n_pos == 0:
        return 1.0
    return max(1.0, n_neg / n_pos)


def build_stage1_pipeline(model_name: str, X_train: pd.DataFrame, y_train: np.ndarray) -> Pipeline:
    pre = build_preprocessor(X_train)
    model_key = model_name.lower()

    if model_key == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    if model_key == "svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=10000,
            random_state=42,
        )
        return Pipeline(
            [
                ("preprocessor", pre),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", clf),
            ]
        )

    if model_key == "xgboost":
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.3,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=compute_scale_pos_weight(y_train),
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    raise ValueError("Unsupported model. Use: randomforest, svm, xgboost")


def build_stage2_pipeline(model_name: str, X_train: pd.DataFrame) -> Pipeline:
    pre = build_preprocessor(X_train)
    model_key = model_name.lower()

    if model_key == "randomforest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=8,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    if model_key == "svm":
        clf = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=False,
            max_iter=10000,
            random_state=42,
        )
        return Pipeline(
            [
                ("preprocessor", pre),
                ("scaler", StandardScaler(with_mean=False)),
                ("classifier", clf),
            ]
        )

    if model_key == "xgboost":
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=1.5,
            eval_metric="mlogloss",
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("preprocessor", pre), ("classifier", clf)])

    raise ValueError("Unsupported model. Use: randomforest, svm, xgboost")


def fit_with_sample_weights(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    classes, counts = np.unique(y, return_counts=True)
    class_weights = {cls: (len(y) / (len(classes) * cnt)) for cls, cnt in zip(classes, counts)}
    sample_weight = np.array([class_weights[val] for val in y])
    pipeline.fit(X, y, classifier__sample_weight=sample_weight)
    return pipeline


def save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def save_label_encoder(path: str, encoder: LabelEncoder) -> None:
    joblib.dump(encoder, path)


def load_label_encoder(path: str) -> LabelEncoder:
    return joblib.load(path)


def summarize_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    fpr_normal = fp / (fp + tn) if (fp + tn) else 0.0
    attack_recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "fpr_normal": float(fpr_normal),
        "attack_recall": float(attack_recall),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def summarize_multiclass(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> Dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=list(class_names),
            zero_division=0,
            output_dict=True,
        ),
    }


def split_train_val(df: pd.DataFrame, val_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    y = to_binary_labels(df["label"])
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
