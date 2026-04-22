# Cascade Step-by-Step Commands

This flow is reusable for all three models by changing only the --model value.

Allowed model values:
- randomforest
- svm
- xgboost

## 0) Activate environment and move to project root

```powershell
cd C:\Users\OWAIS\MyStuff\Labs\Sem6MiniML\CyberSentinel
```

## 1) Train Stage 1 binary model

```powershell
python -m src.train_stage1_binary --model xgboost --train-path data/train/KDDTrain+.csv
```

## 2) Train Stage 2 attack-category model

```powershell
python -m src.train_stage2_category --model xgboost --train-path data/train/KDDTrain+.csv
```

## 3) Calibrate probabilities (mandatory)

```powershell
python -m src.calibrate_models --model xgboost --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
```

Use isotonic instead of sigmoid if needed:

```powershell
python -m src.calibrate_models --model xgboost --stage both --method isotonic --train-path data/train/KDDTrain+.csv
```

## 4) Tune cascade thresholds on validation split

```powershell
python -m src.threshold_tuning --model xgboost --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
```

## 5) Run cascade inference on a test file

```powershell
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv
```

Unified entrypoint (recommended):

```powershell
python -m src.predict --pipeline cascade --algorithm xgboost --test test21
```

Optionally set output file:

```powershell
python -m src.infer_cascade --model xgboost --input-csv data/test/KDDTest-21.csv --output-csv data/test/KDDTest-21_cascade_xgboost.csv
```

## 6) Switch model name only

To run the same pipeline for RF or SVM, replace only the model value.

Examples:

```powershell
python -m src.train_stage1_binary --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model randomforest --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model randomforest --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.threshold_tuning --model randomforest --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.infer_cascade --model randomforest --input-csv data/test/KDDTest-21.csv
```

```powershell
python -m src.train_stage1_binary --model svm --train-path data/train/KDDTrain+.csv
python -m src.train_stage2_category --model svm --train-path data/train/KDDTrain+.csv
python -m src.calibrate_models --model svm --stage both --method sigmoid --train-path data/train/KDDTrain+.csv
python -m src.threshold_tuning --model svm --train-path data/train/KDDTrain+.csv --min-attack-recall 0.90
python -m src.infer_cascade --model svm --input-csv data/test/KDDTest-21.csv
```

## 7) Where artifacts are saved

For each model, artifacts are saved under:

- model/cascade/<model_name>/stage1_model.pkl
- model/cascade/<model_name>/stage2_model.pkl
- model/cascade/<model_name>/stage1_calibrated.pkl
- model/cascade/<model_name>/stage2_calibrated.pkl
- model/cascade/<model_name>/stage2_label_encoder.pkl
- model/cascade/<model_name>/cascade_config.json
- model/cascade/<model_name>/threshold_leaderboard_top20.json
- model/cascade/<model_name>/stage1_metrics.json
- model/cascade/<model_name>/stage2_metrics.json
- model/cascade/<model_name>/calibration_metrics.json
