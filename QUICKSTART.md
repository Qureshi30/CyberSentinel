# 🚀 Quick Start Guide

Get CyberSentinel running in 5 minutes!

## Step 1: Install Dependencies (1 minute)

Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

## Step 2: Get Training Data (2 minutes)

1. Download NSL-KDD dataset from: https://www.unb.ca/cic/datasets/nsl.html
2. Extract **KDDTrain+.csv**
3. Place it in `data/train/` directory

## Step 3: Train the Model (1 minute)

Run the training script:

```bash
python src/train_model.py
```

Wait for training to complete. You'll see accuracy metrics when done.

## Step 4: Start the Web App (30 seconds)

Launch the Flask application:

```bash
python app.py
```

## Step 5: Use the Classifier (30 seconds)

1. Open browser: http://127.0.0.1:5000
2. Upload a CSV file with network traffic data
3. Click "Classify Traffic"
4. View your results!

---

## Troubleshooting

**Problem:** Model not found error  
**Solution:** Make sure you completed Step 3 (training)

**Problem:** Training data not found  
**Solution:** Ensure KDDTrain+.csv is in `data/train/` directory

**Problem:** CSV upload error  
**Solution:** Verify your CSV has the same features as training data

---

## Need Help?

Check the full [README.md](README.md) for detailed documentation.
