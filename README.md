# 🛡️ CyberSentinel - Network Traffic Classifier

A basic machine learning demonstration project that classifies network traffic as either **Threat** or **Harmless** using the NSL-KDD dataset. This project uses a Random Forest classifier to detect potential intrusions in network traffic patterns.

## 📋 Description

CyberSentinel is a simple intrusion detection system that demonstrates how machine learning can be applied to cybersecurity. The system trains on the NSL-KDD dataset and provides a web interface where users can upload CSV files containing network traffic data to get real-time classification results.

**Note:** This is for demonstration purposes only and should not be used in production environments.

## 📁 Project Structure

```
CyberSentinel/
│
├── data/
│   ├── train/
│   │   └── KDDTrain+.csv          # Training dataset (NSL-KDD)
│   └── test/
│       └── sample_input.csv       # Uploaded CSV files for prediction
│
├── model/
│   └── intrusion_model.pkl        # Trained ML model (generated after training)
│
├── src/
│   ├── train_model.py             # Script to train the model
│   ├── predict.py                 # Script to make predictions
│   └── preprocess.py              # Data preprocessing utilities
│
├── app.py                          # Flask web application
│
├── templates/
│   └── index.html                  # Web interface (single page)
│
├── requirements.txt                # Python dependencies
│
└── README.md                       # Project documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Zaheen-Siddiqui/CyberSentinel.git
   cd CyberSentinel
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the training data**
   - Download the NSL-KDD dataset (KDDTrain+.csv)
   - Place it in the `data/train/` directory
   - The CSV should contain network traffic features with a label column

### Usage

#### Step 1: Train the Model

Before using the classifier, you need to train the model with the NSL-KDD dataset:

```bash
python src/train_model.py
```

This will:
- Load the training data from `data/train/KDDTrain+.csv`
- Preprocess and clean the data
- Convert labels to binary classification (normal → Harmless, others → Threat)
- Train a Random Forest classifier
- Save the trained model to `model/intrusion_model.pkl`
- Display training accuracy and metrics

#### Step 2: Run the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://127.0.0.1:5000
```

#### Step 3: Classify Network Traffic

1. On the web page, click "Choose File" and select a CSV file containing network traffic data
2. Click "Classify Traffic"
3. View the results showing:
   - Total records analyzed
   - Number of threats detected
   - Number of harmless connections
   - Detailed table of predictions

### Command-Line Prediction (Optional)

You can also run predictions from the command line:

```bash
python src/predict.py data/test/sample_input.csv
```

This will output predictions to a new CSV file with "_predictions" suffix.

## 📊 CSV File Requirements

Your input CSV file should:
- Contain the same features as the training data
- Have one row per network connection
- Include numerical and/or categorical features
- **Not require** a label column (the model will predict this)

Example features might include:
- Protocol type
- Service
- Flag
- Duration
- Bytes sent/received
- etc.

## 🔍 How It Works

1. **Training Phase:**
   - Loads NSL-KDD training dataset
   - Cleans and preprocesses data
   - Encodes categorical features
   - Converts multi-class labels to binary (Threat/Harmless)
   - Trains Random Forest classifier with 100 trees
   - Saves model and encoders for later use

2. **Prediction Phase:**
   - User uploads a CSV file via web interface
   - System loads the trained model
   - Preprocesses the uploaded data using same encoders
   - Makes predictions for each row
   - Displays results with statistics

## 🛠️ Technologies Used

- **Python** - Programming language
- **Flask** - Web framework
- **scikit-learn** - Machine learning library
- **pandas** - Data manipulation
- **joblib** - Model serialization
- **HTML** - Web interface

## ⚠️ Limitations

- This is a basic demonstration project
- Not suitable for production cybersecurity systems
- Model accuracy depends on training data quality
- No user authentication or session management
- Single-page interface only

## 📝 License

This project is for educational purposes only.

## 👤 Author

Zaheen Siddiqui

## 🤝 Contributing

This is a demonstration project and is not actively maintained for contributions.

---

**Disclaimer:** This tool is for educational and demonstration purposes only. Do not use in production environments or rely on it for actual security decisions.
