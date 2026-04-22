# CyberSentinel Two-Stage Pipeline Implementation Plan (Updated)

## 1) Proposed Pipeline (As Requested)

INPUT  
-> Stage 1: Binary Classifier (NORMAL vs ATTACK, RF/XGBoost/SVM)  
-> If NORMAL: output NORMAL  
-> If ATTACK: Stage 2 Attack Category Classifier  
-> Confidence and consistency checks  
-> Conditional false-positive correction  
-> Final output: NORMAL or Attack Category (optional mapping to specific attack type)

This plan keeps your current algorithm families only (Random Forest, SVM, XGBoost).

## 2) Critical Design Rules (Mandatory)

### Rule A: Probability calibration is mandatory for both stages
- Stage 1 and Stage 2 probabilities are decision-critical (`p_attack`, `p_cat`).
- Uncalibrated probabilities make thresholding and confidence gating unreliable.
- Calibrate using:
  - Platt scaling (`sigmoid`) or
  - Isotonic regression
- Apply calibration on a validation/calibration split only.

Model notes:
- Random Forest: tends to be overconfident, calibration strongly recommended.
- SVM: requires calibration for reliable probabilities.
- XGBoost: often better probability ranking but still requires validation and calibration check.

### Rule B: Conditional revert-to-NORMAL (do not use unconditional revert)
- Do not revert to NORMAL using Stage 2 confidence alone.
- Revert to NORMAL only when BOTH are true:
  1. Stage 1 attack confidence is weak/moderate
  2. Stage 2 confidence is low

Recommended logic:
- If `p_attack >= t_stage1_strong`, keep ATTACK path even when Stage 2 confidence is modest.
- If `t_stage1_low <= p_attack < t_stage1_strong` and Stage 2 confidence is low, allow NORMAL reversion.

Example policy:
- `p_attack = 0.55`, `p_cat = 0.30` -> NORMAL
- `p_attack = 0.92`, `p_cat = 0.40` -> KEEP ATTACK

### Rule C: Explicit class imbalance handling in both stages
- Stage 1 binary:
  - Random Forest/SVM: `class_weight="balanced"` (or tuned weights)
  - XGBoost: `scale_pos_weight` from class ratio
- Stage 2 multiclass:
  - Use per-class weighting to protect rare classes (especially R2L/U2R)
  - Track macro-F1 and per-class recall during tuning

### Rule D: Strict feature transformation consistency
- Stage 1 and Stage 2 must share identical feature cleaning and transformations:
  - same dropped columns
  - same categorical handling and one-hot strategy
  - same numeric transformations
  - same column ordering and schema checks
- Enforce via shared preprocessing module and persisted transformer artifacts.

### Rule E: Leakage-safe data protocol
- Strict sequence:
  1. Split full dataset into train/validation/test
  2. Build Stage 2 training data by filtering ATTACK only inside train split
  3. Build Stage 2 validation data by filtering ATTACK only inside validation split
- Never use test labels for threshold or calibration tuning.

### Rule F: Baseline vs cascade evaluation protocol
- Compare baseline and cascade under identical conditions:
  - same test set
  - same preprocessing
  - same metric definitions
  - same reporting format

## 3) Expected Effects on the Three Models

### Random Forest
- Stage 1 effect:
  - Usually strong attack recall when threshold is lowered.
  - May over-flag borderline NORMAL if threshold too low.
- Stage 2 effect:
  - Good multiclass baseline for DoS/Probe and acceptable coarse category separation.
  - Calibration is important before confidence gating.
- Net expected behavior:
  - Strong candidate if calibrated and guarded with conditional revert logic.

### XGBoost
- Stage 1 effect:
  - Strong first-stage gate candidate with good precision-recall tradeoff.
  - Works well with threshold tuning and imbalance controls.
- Stage 2 effect:
  - Strong multiclass performance when rare class handling is explicit.
- Net expected behavior:
  - Likely best end-to-end baseline in your current stack.

### SVM
- Stage 1 effect:
  - Can generalize well; probability quality depends on calibration quality.
- Stage 2 effect:
  - Feasible but can be less practical for many imbalanced classes.
- Net expected behavior:
  - Good comparison model; calibration and class weighting are essential.

## 4) System-Level Effects You Should Expect

- Final precision likely increases due to better false-positive control.
- Final attack recall can drop if gates are too strict.
- Error profile shifts:
  - fewer false ATTACK alarms
  - potential increase in missed rare attacks
- Operational gain:
  - decision-path transparency improves explainability.

## 5) Data and Label Preparation

### Stage 1 labels
- `y_binary`: NORMAL = 0, ATTACK = 1.

### Stage 2 labels
- Train only on ATTACK samples from the relevant split.
- Build `attack_category` using fixed mapping from original attack labels.
- Recommended categories: DoS, Probe, R2L, U2R.

### Split and filtering policy (strict)
- Split once at record level into train/validation/test.
- Filter ATTACK-only subsets only after split and only within each split.
- Keep test set untouched for final evaluation only.

## 6) Training Plan (Python Script Approach)

### File additions (suggested)
- `src/train_stage1_binary.py`
- `src/train_stage2_category.py`
- `src/calibrate_models.py`
- `src/threshold_tuning.py`
- `src/infer_cascade.py`
- `config/cascade_config.yaml` (or JSON)

### Shared preprocessing contract
- Centralize in one module, for example `src/preprocess.py` + strict schema checks.
- Persist and reuse the same transformation artifacts between stages where applicable.

### Stage 1 training
1. Train binary model (`normal` vs `attack`) with explicit imbalance handling.
2. Calibrate probabilities on validation/calibration set.
3. Tune binary threshold using target metric and constraints.
4. Save model + calibrator + threshold + metrics.

### Stage 2 training
1. Train on ATTACK-only rows from train split.
2. Apply explicit multiclass imbalance handling.
3. Calibrate Stage 2 probabilities.
4. Evaluate macro-F1 and per-class recall.
5. Save model + calibrator + metrics.

### Confidence gate tuning (validation only)
1. Run full cascade over validation split.
2. Compute Stage 2 confidence features:
  - top-1 probability (`top1`)
  - top-2 probability (`top2`)
  - confidence margin (`margin = top1 - top2`)
3. Sweep thresholds for:
  - Stage 1 low/strong boundaries
  - Stage 2 confidence threshold
  - Stage 2 margin threshold
4. Select thresholds under constraints:
  - preserve minimum ATTACK recall
  - minimize NORMAL false positive rate
5. Save chosen thresholds to config.

## 7) Inference Logic (Cascade)

Pseudo-flow:

1. Preprocess input with shared transformation contract.
2. Stage 1 calibrated probability -> `p_attack`.
3. If `p_attack < t_stage1_low`: output NORMAL.
4. Else run Stage 2 calibrated category model -> class probabilities.
5. Compute:
  - `top1 = max(category_probs)`
  - `top2 = second_max(category_probs)`
  - `margin = top1 - top2`
6. Apply conditional gating:
  - if `p_attack >= t_stage1_strong`: keep ATTACK (do not revert to NORMAL)
  - else if (`top1 < t_stage2_conf`) or (`margin < t_stage2_margin`): revert to NORMAL
  - else output predicted attack category
7. Emit decision path and confidence fields.

## 8) Evaluation Plan (Must-Have)

Report at three levels with fixed definitions.

### Stage 1 metrics
- Binary confusion matrix
- Precision, Recall, F1, F2
- PR-AUC (optional)
- Calibration quality (Brier score, reliability curve)

### Stage 2 metrics (true ATTACK subset)
- Macro-F1
- Per-category precision/recall
- Confusion matrix by attack category
- Calibration quality per class (or global multiclass calibration diagnostics)

### End-to-end cascade metrics
- Final multiclass metrics over NORMAL + categories
- Attack missed rate
- Decision-path statistics
- NORMAL-specific false positive rate (mandatory):
  - `FPR_normal = FP_normal / (FP_normal + TN_normal)`

## 9) Integration Into Current Project

### Existing integration points
- Main prediction path in `src/predict.py`
- Dashboard metrics in `app.py`

### Integration changes
- Add prediction mode: `--pipeline cascade`
- Keep current single-model mode for baseline
- Extend output CSV with:
  - `stage1_p_attack`
  - `stage1_decision`
  - `stage2_pred_category`
  - `stage2_top1_conf`
  - `stage2_margin`
  - `final_prediction`
  - `decision_path`

## 10) Notebook-Friendly Implementation Plan

Suggested notebook: `attack_pipeline_experiments.ipynb`

Cell flow:
1. Data load + shared preprocessing contract.
2. Split train/validation/test (before Stage 2 filtering).
3. Build Stage 1 binary targets and Stage 2 category targets.
4. Train Stage 1 with imbalance handling.
5. Calibrate Stage 1 probabilities.
6. Train Stage 2 on ATTACK-only train subset with imbalance handling.
7. Calibrate Stage 2 probabilities.
8. Threshold and margin sweeps on validation.
9. End-to-end cascade evaluation and baseline comparison.
10. Export config and artifacts.

Use notebook for experimentation only; keep production inference in `src/` modules.

## 11) Rollout Strategy

1. Implement cascade in parallel with existing one-stage flow.
2. Run baseline and cascade on same test sets (`KDDTest-21`, `KDDTest+`).
3. Approve thresholds only if constraints are met:
  - minimum attack recall
  - acceptable NORMAL FPR
4. Promote cascade to default after stable repeated runs.

## 12) Risks and Controls

- Risk: rare attacks (R2L/U2R) reverted to NORMAL.
  - Control: conditional revert rule + class weighting + recall constraints.
- Risk: uncalibrated probabilities break threshold meaning.
  - Control: mandatory calibration for both stages.
- Risk: leakage during Stage 2 data preparation.
  - Control: split-first, filter-second protocol.
- Risk: metric inflation due to unfair baseline comparison.
  - Control: strict same-test, same-preprocess, same-metrics evaluation protocol.

## 13) Minimal First Deliverable (Recommended)

- Stage 1: XGBoost binary model + calibration + tuned low/strong thresholds.
- Stage 2: XGBoost multiclass category model + calibration.
- Conditional confidence gate using `top1` and `margin`.
- CLI inference command + CSV decision-path columns.
- Baseline vs cascade report on identical test protocol.

This gives a fast, high-quality baseline while preserving an easy path for RF and SVM variants.
