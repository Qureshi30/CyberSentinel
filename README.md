# 🛡️ CyberSentinel - Multi-Model Network Intrusion Detection System

An advanced machine learning project that classifies network traffic in a **two-stage pipeline**:

1. **Stage 1: Binary Classification** - Normal vs Attack
2. **Stage 2: Attack Classification** - DoS, Probe, R2L, U2R, and any trained attack categories

CyberSentinel supports RandomForest, SVM, and XGBoost models trained on the NSL-KDD dataset for comprehensive network intrusion detection.

## 📋 Description

CyberSentinel is a sophisticated intrusion detection system that demonstrates how multiple machine learning algorithms can be applied to cybersecurity. The system trains three different models and allows comparison of their performance for optimal intrusion detection.

**Supported Models:**
- 🌲 **Random Forest** - Ensemble decision trees for robust classification
- 🎯 **Support Vector Machine (SVM)** - Kernel-based classification with RBF
- 🚀 **XGBoost** - Gradient boosting for high-performance prediction

**Dashboard Output:**
- Stage 1 result: `Normal` or `Attack`
- Stage 2 result: attack category breakdown when an attack is detected
- Live model comparison for all three models
- Dark and light dashboard support

## 📁 Project Structure

```
CyberSentinel/
│
├── data/
│   ├── train/
│   │   ├── KDDTrain+.csv              # Full training dataset
│   │   └── KDDTrain+_20Percent.csv    # 20% subset for faster training
│   │
│   └── test/
│       ├── KDDTest-21.csv             # Primary test dataset
│       ├── KDDTest+.csv               # Extended test dataset
│       └── *_predictions_*.csv        # Generated prediction results
│
├── model/
│   ├── randomforest.pkl               # Trained Random Forest model
│   ├── svm.pkl                        # Trained SVM model
│   ├── xgboost.pkl                    # Trained XGBoost model
│   ├── randomForest.py                # Individual RF training script
│   ├── svm.py                         # Individual SVM training script
│   └── xgboost.py                     # Individual XGBoost training script
│
├── src/
│   ├── train_model.py                 # Multi-model training script
│   ├── predict.py                     # Multi-model prediction script
│   └── preprocess.py                  # Data preprocessing utilities
│
├── templates/
│   └── index.html                     # Web interface
│
├── app.py                             # Flask web application
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## 🚀 Quick Start Guide

Get CyberSentinel running in 5 minutes!

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Windows/Linux/macOS

### Step 1: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Or install individually if needed
pip install --only-binary=all Flask pandas scikit-learn joblib numpy xgboost
```

### Step 2: Prepare Training Data

1. Download NSL-KDD dataset from: https://www.unb.ca/cic/datasets/nsl.html
2. Extract **KDDTrain+.csv** and **KDDTest-21.csv**, **KDDTest+.csv**
3. Place training files in `data/train/` directory
4. Place test files in `data/test/` directory

### Step 3: Train Models

**Train a single model:**
```bash
python src/train_model.py --algorithms randomforest
```

**Train all three models (recommended):**
```bash
python src/train_model.py --algorithms randomforest svm xgboost
```

**Example Output:**
```
============================================================
INTRUSION DETECTION MODEL TRAINING
Training algorithms: randomforest, svm, xgboost
============================================================

[1/4] Loading training data...
   ✓ Loaded 125973 records with 43 columns

[2/4] Cleaning data...
   ✓ Data cleaned. 125973 records remaining

[3/4] Preparing features and labels...
   ✓ Features: 41 columns, 125973 samples
   ✓ Labels: 58630 attacks, 67343 normal

[4/4] Training models...
   ✓ RANDOMFOREST Accuracy: 99.91%
   ✓ SVM Accuracy: 99.16%
   ✓ XGBOOST Accuracy: 99.91%

BEST MODEL: XGBOOST (99.91% accuracy)
```

### Step 4: Test Model Performance

**Test with predefined datasets:**
```bash
# Test on KDDTest-21.csv
python src/predict.py --test test21 --algorithm xgboost

# Test on KDDTest+.csv  
python src/predict.py --test testplus --algorithm randomforest
```

**Test with custom CSV file:**
```bash
python src/predict.py your_data.csv --algorithm svm
```

### Step 5: Launch Web Application / Dashboard

```bash
python app.py
```
Open browser: http://127.0.0.1:5000

If you want the React dashboard, run:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal, usually http://127.0.0.1:5173

## 📊 Detailed Usage Instructions

### Training Models

The training script supports flexible model selection:

```bash
# Train individual models
python src/train_model.py --algorithms randomforest
python src/train_model.py --algorithms svm  
python src/train_model.py --algorithms xgboost

# Train multiple models
python src/train_model.py --algorithms randomforest svm
python src/train_model.py --algorithms svm xgboost
python src/train_model.py --algorithms randomforest svm xgboost
```

**Training Features:**
- ✅ Automatic data preprocessing and cleaning
- ✅ One-hot encoding for categorical variables
- ✅ Binary classification (normal=0, attack=1)
- ✅ Stratified train-test split (80/20)
- ✅ Model-specific hyperparameter optimization
- ✅ Performance evaluation and comparison
- ✅ Automatic model saving (.pkl files)

### Making Predictions

The prediction script offers multiple input options:

**Option 1: Predefined Test Datasets**
```bash
# Use KDDTest-21.csv (11,850 records)
python src/predict.py --test test21 --algorithm xgboost

# Use KDDTest+.csv (22,544 records)  
python src/predict.py --test testplus --algorithm randomforest
```

**Option 2: Custom CSV Files**
```bash
python src/predict.py path/to/your/data.csv --algorithm svm
```

**Option 3: List Available Models**
```bash
python src/predict.py --list-models
```

**Prediction Output:**
```
Using test dataset: data\test\KDDTest-21.csv
Loading data from data\test\KDDTest-21.csv...
Loaded 11850 records
Loading XGBOOST model...
Making predictions...
   ✓ Predictions completed for 11850 records
   ✓ Accuracy: 60.03%

Prediction Summary:
  Total: 11850
  Attack: 5499 (46.4%)
  Normal: 6351 (53.6%)

Predictions saved to: data\test\KDDTest-21_predictions_xgboost.csv
```

### CSV File Format

Your input CSV must contain these 41 network traffic features:

**Basic Features:**
- `duration`, `protocol_type`, `service`, `flag`
- `src_bytes`, `dst_bytes`, `land`, `wrong_fragment`, `urgent`

**Content Features:**
- `hot`, `num_failed_logins`, `logged_in`, `num_compromised`
- `root_shell`, `su_attempted`, `num_root`, `num_file_creations`

**Traffic Features:**
- `count`, `srv_count`, `serror_rate`, `srv_serror_rate`
- `rerror_rate`, `srv_rerror_rate`, `same_srv_rate`, `diff_srv_rate`

**Host Features:**
- `dst_host_count`, `dst_host_srv_count`, `dst_host_same_srv_rate`
- `dst_host_diff_srv_rate`, `dst_host_same_src_port_rate`

**Optional:**
- `label` (for accuracy calculation)
- `difficulty` (automatically removed)

## 📈 Model Performance Comparison

**Training Performance (KDDTrain+.csv):**
| Model | Accuracy | Training Time | Model Size |
|-------|----------|---------------|------------|
| RandomForest | 99.91% | Fast | Medium |
| SVM | 99.16% | Slow | Small |
| XGBoost | 99.91% | Medium | Large |

**Test Performance:**

*KDDTest-21.csv (11,850 records):*
| Model | Accuracy | Attack Detection | Normal Detection |
|-------|----------|------------------|------------------|
| RandomForest | 56.28% | 42.6% | 57.4% |
| SVM | 59.12% | 53.0% | 47.0% |
| **XGBoost** | **60.03%** | **46.4%** | **53.6%** |

*KDDTest+.csv (22,544 records):*
| Model | Accuracy | Attack Detection | Normal Detection |
|-------|----------|------------------|------------------|
| **XGBoost** | **78.99%** | **38.3%** | **61.7%** |

**Recommendations:**
- 🏆 **XGBoost**: Best overall performance and generalization
- 🌲 **RandomForest**: Good balance of speed and accuracy
- 🎯 **SVM**: Good for smaller datasets but slower training

## 🛠️ Web Application

Launch the Flask web interface for easy file uploads:

```bash
python app.py
```

**Features:**
- 📁 Drag-and-drop CSV file upload
- 🔄 Real-time prediction processing
- 📊 Visual results with statistics
- 📋 Downloadable prediction results
- 🎛️ Model selection interface
- 🧭 Explicit two-stage output view for Stage 1 and Stage 2
- 🌗 Light and dark theme support in the React dashboard

## 🔧 Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'xgboost'`
```bash
Solution: pip install xgboost
```

**Problem:** `FileNotFoundError: Model not found`
```bash
Solution: Train models first using:
python src/train_model.py --algorithms randomforest svm xgboost
```

**Problem:** `Error loading CSV file`
```bash
Solution: Ensure CSV has required columns and proper format
Check data/test/KDDTest-21.csv for reference
```

**Problem:** Low prediction accuracy
```bash
Solution: Ensure test data format matches training data
Verify categorical columns: protocol_type, service, flag
```

**Problem:** Long training time
```bash
Solution: Use subset training data:
Place KDDTrain+_20Percent.csv in data/train/ directory
```

## 🔬 Technical Details

### Model Specifications

**Random Forest:**
- 100 estimators with balanced class weights
- Gini impurity criterion
- Bootstrap sampling enabled
- Random state: 42

**SVM:**
- RBF kernel with gamma='scale'
- Balanced class weights
- StandardScaler preprocessing
- C=1.0 (default)

**XGBoost:**
- 200 estimators, max_depth=6
- Learning rate: 0.1
- Subsample: 0.8, colsample_bytree: 0.8
- Binary logistic objective

### Data Preprocessing Pipeline

1. **Load CSV** → pandas DataFrame
2. **Clean Data** → handle missing values, remove duplicates
3. **Drop Columns** → remove 'difficulty' if present
4. **Encode Labels** → normal=0, attacks=1
5. **One-Hot Encoding** → categorical features (protocol_type, service, flag)
6. **Feature Scaling** → StandardScaler for SVM only
7. **Train-Test Split** → 80/20 stratified split

## 📝 File Outputs

**After Training:**
- `model/randomforest.pkl` → Trained Random Forest model
- `model/svm.pkl` → Trained SVM model  
- `model/xgboost.pkl` → Trained XGBoost model

**After Prediction:**
- `[input_file]_predictions_[algorithm].csv` → Results with predictions
- Cascade output columns include:
   - `stage1_p_attack`
   - `stage1_decision`
   - `stage2_pred_category`
   - `stage2_top1_conf`
   - `stage2_margin`
   - `final_prediction`
   - `decision_path`

## ⚠️ Important Notes

- 🔒 **Educational Purpose**: Not for production security systems
- 📊 **Data Quality**: Results depend on input data quality  
- 🧠 **Model Drift**: Retrain periodically with new data
- ⚡ **Performance**: XGBoost recommended for best accuracy
- 💾 **Storage**: Models require ~10-50MB disk space each

## 🤝 Contributing

This is an educational project. For improvements:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## 👤 Author
**Zaheen Siddiqui** - Network Security & Machine Learning
GitHub: [Zaheen-Siddiqui](https://github.com/Zaheen-Siddiqui)

**Prem Hanchate** - Network Security & Machine Learning
GitHub: [Prem-Hanchate](https://github.com/Prem-Hanchate)

## 📄 License

Educational use only. Not for commercial deployment.

---

**⚠️ Disclaimer:** This tool demonstrates ML concepts for cybersecurity education. Do not rely on it for actual network security decisions or production environments.
