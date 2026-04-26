"""
Flask Web Application
Simple web interface for network traffic classification
"""

import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from src.predict import predict_from_csv, list_available_models, list_available_cascade_models
from src.preprocess import clean_data

app = Flask(__name__)

REALTIME_INPUT_FILE = os.path.join("data", "test", "realtime_input.csv")
ATTACK_CATEGORY_ORDER = ["DoS", "Probe", "R2L", "U2R", "Uncategorized"]

MODEL_DASHBOARD_SOURCES = {
    "xgboost": {
        "display_name": "XGBoost",
        "color": "#2f80ed",
        "model_paths": [
            os.path.join("model", "xgboost", "xgboost_regularised.pkl"),
            os.path.join("model", "xgboost.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "KDDTest+_predictions_xgboost.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest+_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest+_predictions_xgboost.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_xgboost.csv"),
                os.path.join("data", "test", "KDDTest-21_cascade_xgboost.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions_xgboost.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_xgboost.csv"),
                os.path.join("data", "test", "KDDTest-21_cascade_xgboost.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions_xgboost.csv"),
            ],
        },
    },
    "random_forest": {
        "display_name": "Random Forest",
        "color": "#4c9b1d",
        "model_paths": [
            os.path.join("model", "random_forest.pkl"),
            os.path.join("model", "randomforest.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "KDDTest+_predictions_randomforest.csv"),
                os.path.join("data", "test", "random_forest", "KDDTest+_predictions.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_randomforest.csv"),
                os.path.join("data", "test", "random_forest", "KDDTest-21_predictions.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_randomforest.csv"),
                os.path.join("data", "test", "random_forest", "KDDTest-21_predictions.csv"),
            ],
        },
    },
    "svm": {
        "display_name": "SVM",
        "color": "#a26411",
        "model_paths": [
            os.path.join("model", "svm.pkl"),
        ],
        "prediction_paths": {
            "acc_test_plus": [
                os.path.join("data", "test", "KDDTest+_predictions_svm.csv"),
                os.path.join("data", "test", "svm", "KDDTest+_predictions.csv"),
            ],
            "acc_test_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_svm.csv"),
                os.path.join("data", "test", "svm", "KDDTest-21_predictions.csv"),
            ],
            "attack_recall_21": [
                os.path.join("data", "test", "KDDTest-21_predictions_svm.csv"),
                os.path.join("data", "test", "svm", "KDDTest-21_predictions.csv"),
            ],
        },
    },
}

# Configure upload folder
UPLOAD_FOLDER = os.path.join('data', 'test')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def _first_existing_path(path_list):
    for path in path_list:
        if os.path.exists(path):
            return path
    return None


def _find_column(df, candidates):
    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def _to_binary_label(value):
    text = str(value).strip().lower()
    return 0 if text == "normal" else 1


def _compute_prediction_file_metrics(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return {}

    label_col = _find_column(df, ["label", "actual_label", "class"])
    pred_col = _find_column(df, ["final_prediction", "prediction", "pred", "predicted_label"])

    if label_col is None or pred_col is None or df.empty:
        return {}

    y_true = df[label_col].apply(_to_binary_label)
    y_pred = df[pred_col].apply(_to_binary_label)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0) * 100.0
    recall = recall_score(y_true, y_pred, zero_division=0) * 100.0
    accuracy = accuracy_score(y_true, y_pred) * 100.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    fnr = (fn / (fn + tp) * 100.0) if (fn + tp) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0) * 100.0
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0) * 100.0

    return {
        "accuracy": round(float(accuracy), 2),
        "precision": round(float(precision), 2),
        "recall": round(float(recall), 2),
        "specificity": round(float(specificity), 2),
        "false_positive_rate": round(float(fpr), 2),
        "false_negative_rate": round(float(fnr), 2),
        "f1_score": round(float(f1), 2),
        "f2_score": round(float(f2), 2),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "support_negative": int(tn + fp),
        "support_positive": int(tp + fn),
    }


def _compute_train_val_gap(model_path):
    train_candidates = [
        os.path.join("data", "train", "KDDTrain+.csv"),
        os.path.join("data", "train", "KDDTrain+_20Percent.csv"),
    ]
    train_path = _first_existing_path(train_candidates)

    if train_path is None or not os.path.exists(model_path):
        return None

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(train_path)
        df = clean_data(df)

        if "label" not in df.columns:
            return None

        if "difficulty" in df.columns:
            df = df.drop(columns=["difficulty"])

        y = df["label"].apply(_to_binary_label)
        X = df.drop(columns=[col for col in ["label", "actual_label", "class"] if col in df.columns])

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        train_acc = accuracy_score(y_train, model.predict(X_train))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        gap = abs(train_acc - val_acc) * 100.0
        return round(float(gap), 2)
    except Exception:
        return None


def _build_dashboard_payload():
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "models": {},
    }

    for key, config in MODEL_DASHBOARD_SOURCES.items():
        metric_values = {
            "acc_test_plus": None,
            "acc_test_21": None,
            "attack_recall_21": None,
            "train_val_gap": None,
            "confusion": {
                "tn": None,
                "fp": None,
                "fn": None,
                "tp": None,
                "precision": None,
                "recall": None,
                "specificity": None,
                "false_positive_rate": None,
                "false_negative_rate": None,
                "f1_score": None,
                "f2_score": None,
                "accuracy": None,
            },
        }

        for metric_key, file_candidates in config["prediction_paths"].items():
            metric_file = _first_existing_path(file_candidates)
            if metric_file is None:
                continue

            calculated = _compute_prediction_file_metrics(metric_file)
            if metric_key in ["acc_test_plus", "acc_test_21"]:
                metric_values[metric_key] = calculated["accuracy"]
            elif metric_key == "attack_recall_21":
                metric_values[metric_key] = calculated.get("recall")

            if metric_key == "acc_test_21" and calculated:
                metric_values["confusion"] = {
                    "tn": calculated.get("tn"),
                    "fp": calculated.get("fp"),
                    "fn": calculated.get("fn"),
                    "tp": calculated.get("tp"),
                    "precision": calculated.get("precision"),
                    "recall": calculated.get("recall"),
                    "specificity": calculated.get("specificity"),
                    "false_positive_rate": calculated.get("false_positive_rate"),
                    "false_negative_rate": calculated.get("false_negative_rate"),
                    "f1_score": calculated.get("f1_score"),
                    "f2_score": calculated.get("f2_score"),
                    "accuracy": calculated.get("accuracy"),
                }

        model_path = _first_existing_path(config["model_paths"])
        if model_path is not None:
            metric_values["train_val_gap"] = _compute_train_val_gap(model_path)

        payload["models"][key] = {
            "display_name": config["display_name"],
            "color": config["color"],
            "metrics": metric_values,
        }

    return payload


def _pick_prediction_column(df):
    return _find_column(df, ["final_prediction", "prediction", "Prediction", "predicted_label", "pred"])


def _build_two_stage_summary(result_df, pred_col):
    final_series = result_df[pred_col].astype(str).str.strip()
    normal_mask = final_series.str.lower() == "normal"

    stage1_col = _find_column(result_df, ["stage1_decision"])
    stage2_col = _find_column(result_df, ["stage2_pred_category"])
    if stage1_col is not None:
        stage1_attack = int(result_df[stage1_col].astype(str).str.strip().str.lower().eq("attack_path").sum())
    else:
        stage1_attack = int((~normal_mask).sum())

    stage1_total = int(len(result_df))
    stage1_normal = int(stage1_total - stage1_attack)

    stage2_source = result_df[stage2_col] if stage2_col is not None else final_series
    stage2_values = stage2_source.astype(str).str.strip()
    stage2_values = stage2_values.where(stage2_values.str.lower() != "normal", other="normal")
    stage2_values = stage2_values.where(stage2_values.str.lower() != "attack", other="Uncategorized")
    stage2_values = stage2_values.where(stage2_values.str.lower() != "", other="Uncategorized")
    attack_categories = stage2_values[~normal_mask]
    raw_counts = attack_categories.value_counts().to_dict() if len(attack_categories) else {}

    ordered = {"DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0, "Uncategorized": 0}
    for key, value in raw_counts.items():
        ordered[key] = int(value)

    return {
        "stage1": {
            "total": stage1_total,
            "normal": stage1_normal,
            "attack": stage1_attack,
        },
        "stage2": {
            "total_attacks": int((~normal_mask).sum()),
            "categories": ordered,
            "mode": "categorized" if stage2_col is not None else "binary_only",
        },
    }


def _build_prediction_payload(result_df, display_limit=100):
    if result_df is None or result_df.empty:
        return {
            "summary": {"total": 0, "normal": 0, "attack": 0, "showing": 0},
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {"total_attacks": 0, "categories": {"DoS": 0, "Probe": 0, "R2L": 0, "U2R": 0, "Uncategorized": 0}, "mode": "binary_only"},
            "rows": [],
        }

    pred_col = _pick_prediction_column(result_df)
    if pred_col is None:
        raise ValueError("Prediction output does not contain a recognizable prediction column")

    pred_series = result_df[pred_col].astype(str).str.strip().str.lower()
    is_normal = pred_series == "normal"

    summary = {
        "total": int(len(result_df)),
        "normal": int(is_normal.sum()),
        "attack": int((~is_normal).sum()),
        "normal_percentage": round(float(is_normal.mean() * 100.0), 2),
        "attack_percentage": round(float((~is_normal).mean() * 100.0), 2),
        "showing": int(min(display_limit, len(result_df))),
    }
    stage_summary = _build_two_stage_summary(result_df, pred_col)

    preferred_cols = [
        pred_col,
        "stage1_p_attack",
        "stage2_top1_conf",
        "stage2_margin",
        "decision_path",
        "model_used",
    ]
    preview_cols = [c for c in preferred_cols if c in result_df.columns]
    if len(preview_cols) < 6:
        fallback = [c for c in result_df.columns if c not in preview_cols]
        preview_cols.extend(fallback[: 6 - len(preview_cols)])

    rows = []
    for idx, (_, row) in enumerate(result_df.head(display_limit).iterrows(), start=1):
        features = {}
        for col in preview_cols:
            value = row[col]
            if isinstance(value, float):
                features[col] = round(float(value), 6)
            else:
                features[col] = value

        rows.append(
            {
                "row_number": idx,
                "prediction": str(row[pred_col]),
                "fields": features,
            }
        )

    return {
        "prediction_column": pred_col,
        "preview_columns": preview_cols,
        "summary": summary,
        "stage1": stage_summary["stage1"],
        "stage2": stage_summary["stage2"],
        "rows": rows,
    }


def _safe_float(value, digits=6):
    try:
        return round(float(value), digits)
    except Exception:
        return None


def _is_normal_label(value):
    return str(value).strip().lower() == "normal"


def _build_realtime_model_payload(result_df, display_limit=30):
    if result_df is None or result_df.empty:
        return {
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {"total_attacks": 0, "categories": {}},
            "preview_rows": [],
        }

    final_col = _find_column(result_df, ["final_prediction", "prediction", "predicted_label", "pred"])
    if final_col is None:
        raise ValueError("Prediction output does not contain a final prediction column")

    stage1_col = _find_column(result_df, ["stage1_decision"])
    stage1_attack_prob_col = _find_column(result_df, ["stage1_p_attack"])
    stage2_cat_col = _find_column(result_df, ["stage2_pred_category"])
    stage2_conf_col = _find_column(result_df, ["stage2_top1_conf"])
    decision_path_col = _find_column(result_df, ["decision_path"])

    final_series = result_df[final_col].astype(str).str.strip()
    normal_mask = final_series.str.lower() == "normal"

    if stage1_col is not None:
        stage1_attack_mask = result_df[stage1_col].astype(str).str.strip().str.lower() == "attack_path"
        stage1_attack = int(stage1_attack_mask.sum())
    else:
        stage1_attack = int((~normal_mask).sum())

    stage1_total = int(len(result_df))
    stage1_normal = int(stage1_total - stage1_attack)

    attack_categories = final_series[~normal_mask]
    stage2_counts = (
        attack_categories.value_counts().sort_index().to_dict() if not attack_categories.empty else {}
    )

    preview_rows = []
    for idx, (_, row) in enumerate(result_df.head(display_limit).iterrows(), start=1):
        final_prediction = str(row[final_col])
        stage2_category = None
        if not _is_normal_label(final_prediction):
            if stage2_cat_col is not None:
                stage2_category = str(row[stage2_cat_col])
            else:
                stage2_category = final_prediction

        preview_rows.append(
            {
                "row_number": idx,
                "stage1_decision": str(row[stage1_col]) if stage1_col else ("attack_path" if not _is_normal_label(final_prediction) else "normal"),
                "stage1_p_attack": _safe_float(row[stage1_attack_prob_col], 6) if stage1_attack_prob_col else None,
                "stage2_category": stage2_category,
                "stage2_confidence": _safe_float(row[stage2_conf_col], 6) if stage2_conf_col else None,
                "final_prediction": final_prediction,
                "decision_path": str(row[decision_path_col]) if decision_path_col else None,
            }
        )

    return {
        "stage1": {
            "total": stage1_total,
            "normal": stage1_normal,
            "attack": stage1_attack,
        },
        "stage2": {
            "total_attacks": int((~normal_mask).sum()),
            "categories": stage2_counts,
        },
        "preview_rows": preview_rows,
    }


def _run_realtime_model_prediction(input_csv_path, algorithm, display_limit=30):
    try:
        result_df = predict_from_csv(input_csv_path, algorithm=algorithm, pipeline="cascade")
        payload = _build_realtime_model_payload(result_df, display_limit=display_limit)
        payload["available"] = True
        return payload
    except FileNotFoundError:
        return {
            "available": False,
            "error": "Cascade artifacts not found for this model. Train stage1/stage2/calibration first.",
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {"total_attacks": 0, "categories": {}},
            "preview_rows": [],
        }
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {"total_attacks": 0, "categories": {}},
            "preview_rows": [],
        }


def _normalize_prediction_to_stage2(value):
    text = str(value).strip()
    low = text.lower()
    if low == "normal":
        return "normal"
    if low == "attack":
        return "Uncategorized"
    return text


def _ordered_attack_categories(counts):
    ordered = {key: int(counts.get(key, 0)) for key in ATTACK_CATEGORY_ORDER}
    extra_keys = sorted([k for k in counts.keys() if k not in ordered])
    for key in extra_keys:
        ordered[key] = int(counts.get(key, 0))
    return ordered


def _has_stage2_categories(result_df):
    if result_df is None or result_df.empty:
        return False

    if "stage2_pred_category" in result_df.columns:
        return True

    pred_col = _pick_prediction_column(result_df)
    if pred_col is None:
        return False

    unique_non_normal = {
        str(v).strip().lower()
        for v in result_df[pred_col].dropna().unique().tolist()
        if str(v).strip().lower() != "normal"
    }
    return len(unique_non_normal) > 0 and unique_non_normal != {"attack"}


def _summarize_result_dataframe(result_df):
    if result_df is None or result_df.empty:
        return {
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {
                "total_attacks": 0,
                "categories": _ordered_attack_categories({}),
                "mode": "binary_only",
            },
        }

    pred_col = _pick_prediction_column(result_df)
    if pred_col is None:
        raise ValueError("No prediction column found in results")

    preds = result_df[pred_col].apply(_normalize_prediction_to_stage2)
    normal_mask = preds.astype(str).str.lower() == "normal"

    stage1 = {
        "total": int(len(result_df)),
        "normal": int(normal_mask.sum()),
        "attack": int((~normal_mask).sum()),
    }

    attack_preds = preds[~normal_mask]
    raw_counts = attack_preds.value_counts().to_dict() if len(attack_preds) else {}
    stage2 = {
        "total_attacks": int(len(attack_preds)),
        "categories": _ordered_attack_categories(raw_counts),
        "mode": "categorized" if _has_stage2_categories(result_df) else "binary_only",
    }

    return {
        "stage1": stage1,
        "stage2": stage2,
    }


def _load_existing_prediction_file(dataset_key, model_key):
    file_map = {
        "kddtest_plus": {
            "xgboost": [
                os.path.join("data", "test", "xgboost", "KDDTest+_predictions.csv"),
                os.path.join("data", "test", "KDDTest+_predictions_xgboost.csv"),
            ],
            "svm": [
                os.path.join("data", "test", "svm", "KDDTest+_predictions.csv"),
                os.path.join("data", "test", "KDDTest+_predictions_svm.csv"),
            ],
            "random_forest": [
                os.path.join("data", "test", "random_forest", "KDDTest+_predictions.csv"),
                os.path.join("data", "test", "KDDTest+_predictions_randomforest.csv"),
            ],
        },
        "kddtest_21": {
            "xgboost": [
                os.path.join("data", "test", "xgboost", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "KDDTest-21_predictions_xgboost.csv"),
                os.path.join("data", "test", "KDDTest-21_cascade_xgboost.csv"),
            ],
            "svm": [
                os.path.join("data", "test", "svm", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "KDDTest-21_predictions_svm.csv"),
            ],
            "random_forest": [
                os.path.join("data", "test", "random_forest", "KDDTest-21_predictions.csv"),
                os.path.join("data", "test", "KDDTest-21_predictions_randomforest.csv"),
            ],
        },
    }

    model_candidates = file_map.get(dataset_key, {}).get(model_key, [])
    path = _first_existing_path(model_candidates)
    if path is None:
        return None, None

    try:
        return pd.read_csv(path), path
    except Exception:
        return None, path


def _dataset_input_path(dataset_key):
    if dataset_key == "kddtest_plus":
        return os.path.join("data", "test", "KDDTest+.csv")
    if dataset_key == "kddtest_21":
        return os.path.join("data", "test", "KDDTest-21.csv")
    raise ValueError("Unsupported dataset key")


def _compute_model_dataset_result(dataset_key, model_key, algorithm_name):
    existing_df, source_path = _load_existing_prediction_file(dataset_key, model_key)
    cascade_ready = algorithm_name in list_available_cascade_models()

    if existing_df is not None:
        if _has_stage2_categories(existing_df):
            summary = _summarize_result_dataframe(existing_df)
            return {
                "available": True,
                "source": source_path,
                "pipeline": "precomputed_file",
                **summary,
            }

        if not cascade_ready:
            summary = _summarize_result_dataframe(existing_df)
            return {
                "available": True,
                "source": source_path,
                "pipeline": "precomputed_file",
                "warning": "Only binary outputs found. Stage 2 categories are shown as Uncategorized.",
                **summary,
            }

    input_path = _dataset_input_path(dataset_key)
    fallback_ready = algorithm_name in list_available_models()

    if not cascade_ready and not fallback_ready:
        return {
            "available": False,
            "source": None,
            "pipeline": None,
            "error": "No trained artifacts found for this model.",
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {
                "total_attacks": 0,
                "categories": _ordered_attack_categories({}),
                "mode": "binary_only",
            },
        }

    pipeline = "cascade" if cascade_ready else "single"
    try:
        result_df = predict_from_csv(input_path, algorithm=algorithm_name, pipeline=pipeline)
        summary = _summarize_result_dataframe(result_df)
        return {
            "available": True,
            "source": input_path,
            "pipeline": pipeline,
            **summary,
        }
    except Exception as exc:
        return {
            "available": False,
            "source": input_path,
            "pipeline": pipeline,
            "error": str(exc),
            "stage1": {"total": 0, "normal": 0, "attack": 0},
            "stage2": {
                "total_attacks": 0,
                "categories": _ordered_attack_categories({}),
                "mode": "binary_only",
            },
        }


@app.route('/')
def index():
    """
    Render the main page
    """
    return render_template('index.html', predictions=None, error=None)


@app.route('/api/dashboard-metrics', methods=['GET'])
def dashboard_metrics():
    """
    Return live dashboard metrics for all models.
    """
    return jsonify(_build_dashboard_payload())


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    Handle CSV upload for in-page predictions and return JSON payload.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400

        algorithm = request.form.get('algorithm', 'xgboost').strip().lower()
        pipeline = request.form.get('pipeline', 'cascade').strip().lower()
        if algorithm not in ['randomforest', 'svm', 'xgboost']:
            return jsonify({"error": "Invalid algorithm. Use randomforest, svm, or xgboost."}), 400
        if pipeline not in ['single', 'cascade']:
            return jsonify({"error": "Invalid pipeline. Use single or cascade."}), 400

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sample_input.csv')
        file.save(file_path)

        result_df = predict_from_csv(file_path, algorithm=algorithm, pipeline=pipeline)
        payload = _build_prediction_payload(result_df)
        payload['algorithm'] = algorithm
        payload['pipeline'] = pipeline
        return jsonify(payload)

    except FileNotFoundError:
        return jsonify({
            "error": "Required model artifacts not found. Train the selected model/pipeline first."
        }), 404
    except Exception as e:
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500


@app.route('/api/realtime/upload', methods=['POST'])
def api_realtime_upload():
    """
    Upload and store latest input data for polling-based inference.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({"error": "Please upload a CSV file"}), 400

        os.makedirs(os.path.dirname(REALTIME_INPUT_FILE), exist_ok=True)
        file.save(REALTIME_INPUT_FILE)

        return jsonify({
            "message": "input uploaded",
            "input_file": REALTIME_INPUT_FILE,
        })
    except Exception as exc:
        return jsonify({"error": f"Failed to upload file: {str(exc)}"}), 500


@app.route('/api/realtime/predict-all', methods=['GET', 'POST'])
def api_realtime_predict_all():
    """
    Run two-stage cascade predictions across all three models and return unified dashboard payload.
    """
    try:
        input_path = REALTIME_INPUT_FILE

        if request.method == 'POST' and 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            if not file.filename.lower().endswith('.csv'):
                return jsonify({"error": "Please upload a CSV file"}), 400

            os.makedirs(os.path.dirname(REALTIME_INPUT_FILE), exist_ok=True)
            file.save(REALTIME_INPUT_FILE)

        if not os.path.exists(input_path):
            return jsonify({
                "error": "No input available. Upload a CSV first using /api/realtime/upload.",
            }), 400

        try:
            display_limit = int(request.args.get('limit', 30))
        except ValueError:
            display_limit = 30
        display_limit = max(1, min(display_limit, 200))

        model_outputs = {
            "xgboost": _run_realtime_model_prediction(input_path, "xgboost", display_limit),
            "svm": _run_realtime_model_prediction(input_path, "svm", display_limit),
            "random_forest": _run_realtime_model_prediction(input_path, "randomforest", display_limit),
        }

        return jsonify(
            {
                "generated_at": pd.Timestamp.now("UTC").isoformat(),
                "input_file": input_path,
                "models": model_outputs,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"prediction failed: {str(exc)}"}), 500


@app.route('/api/dataset-results', methods=['GET'])
def api_dataset_results():
    """
    Return model-wise results for the two built-in datasets in data/test.
    Uses precomputed prediction CSV files when available; otherwise attempts model inference.
    """
    try:
        datasets = {
            "kddtest_plus": {
                "name": "KDDTest+",
                "path": os.path.join("data", "test", "KDDTest+.csv"),
            },
            "kddtest_21": {
                "name": "KDDTest-21",
                "path": os.path.join("data", "test", "KDDTest-21.csv"),
            },
        }

        model_map = {
            "xgboost": "xgboost",
            "svm": "svm",
            "random_forest": "randomforest",
        }

        payload = {
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
            "datasets": {},
        }

        for dataset_key, dataset_meta in datasets.items():
            dataset_result = {
                "name": dataset_meta["name"],
                "path": dataset_meta["path"],
                "models": {},
            }
            for model_key, algo in model_map.items():
                model_result = _compute_model_dataset_result(
                    dataset_key=dataset_key,
                    model_key=model_key,
                    algorithm_name=algo,
                )
                model_result["display_name"] = MODEL_DASHBOARD_SOURCES.get(model_key, {}).get(
                    "display_name",
                    model_key,
                )
                dataset_result["models"][model_key] = model_result

            payload["datasets"][dataset_key] = dataset_result

        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Failed to load dataset results: {str(exc)}"}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle CSV upload and make predictions
    """
    # Kept as backward-compatible alias for form posts.
    return api_predict()


if __name__ == '__main__':
    # Check whether any single-model or cascade artifacts are available.
    available_single = list_available_models()
    available_cascade = list_available_cascade_models()

    if not available_single and not available_cascade:
        print("\n" + "="*60)
        print("WARNING: No trained model artifacts found!")
        print("="*60)
        print("Train at least one pipeline first:")
        print("  Single model:  python src/train_model.py")
        print("  Cascade model: python src/train_stage1_binary.py --model xgboost")
        print("                 python src/train_stage2_category.py --model xgboost")
        print("                 python src/calibrate_models.py --model xgboost")
        print("                 python src/threshold_tuning.py --model xgboost")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("Model artifacts detected")
        print("="*60)
        if available_single:
            print(f"Single-model ready: {', '.join(available_single)}")
        if available_cascade:
            print(f"Cascade-ready: {', '.join(available_cascade)}")
        print("="*60 + "\n")
    
    # Run the Flask app
    print("\n" + "="*60)
    print("CYBERSENTINEL - Network Traffic Classifier")
    print("="*60)
    print("Starting web server...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
