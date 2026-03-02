"""
Model Training Script
Trains multiple ML classifiers on NSL-KDD dataset
Supports: RandomForest, SVM, XGBoost
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Add parent directory to path to import preprocess module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_csv, clean_data


def get_model_pipeline(algorithm='randomforest'):
    """
    Create model pipeline for specified algorithm
    
    Args:
        algorithm: 'randomforest', 'svm', or 'xgboost'
    
    Returns:
        sklearn pipeline with preprocessing and model
    """
    # Define categorical columns
    categorical_cols = ["protocol_type", "service", "flag"]
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    # Select model based on algorithm
    if algorithm.lower() == 'randomforest':
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    
    elif algorithm.lower() == 'svm':
        model = SVC(kernel="rbf", class_weight="balanced", random_state=42)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler(with_mean=False)),  # needed for sparse matrix
            ("classifier", model)
        ])
    
    elif algorithm.lower() == 'xgboost':
        model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: randomforest, svm, xgboost")
    
    return pipeline


def train_intrusion_detection_model(algorithms=['randomforest']):
    """
    Main function to train intrusion detection models
    
    Args:
        algorithms: list of algorithms to train ['randomforest', 'svm', 'xgboost']
    """
    print("=" * 60)
    print("INTRUSION DETECTION MODEL TRAINING")
    print(f"Training algorithms: {', '.join(algorithms)}")
    print("=" * 60)
    
    # Define file paths
    train_data_path = os.path.join('data', 'train', 'KDDTrain+_20Percent.csv')  # Use smaller subset for faster training during development
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        print("Please place KDDTrain+.csv in the data/train/ directory")
        return
    
    # Step 1: Load the training data
    print(f"\n[1/4] Loading training data from {train_data_path}...")
    df = load_csv(train_data_path)
    if df is None:
        return
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Step 2: Clean the data
    print("\n[2/4] Cleaning data...")
    df = clean_data(df)
    print(f"   ✓ Data cleaned. {len(df)} records remaining")
    
    # Step 3: Prepare data for training
    print("\n[3/4] Preparing features and labels...")
    
    # Drop difficulty column if exists
    if "difficulty" in df.columns:
        df = df.drop("difficulty", axis=1)
    
    # Convert label to binary (0 for normal, 1 for attack)
    df["label"] = df["label"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)
    
    # Separate features from labels
    X = df.drop("label", axis=1)
    y = df["label"]
    
    print(f"   ✓ Features: {X.shape[1]} columns, {X.shape[0]} samples")
    print(f"   ✓ Labels: {sum(y)} attacks, {len(y)-sum(y)} normal")
    
    # Step 4: Train models
    print("\n[4/4] Training models...")
    os.makedirs('model', exist_ok=True)
    
    results = {}
    
    for algorithm in algorithms:
        print(f"\n   Training {algorithm.upper()} model...")
        
        try:
            # Get model pipeline
            pipeline = get_model_pipeline(algorithm)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"      ✓ {algorithm.upper()} Accuracy: {accuracy * 100:.2f}%")
            
            # Save model
            model_filename = f"{algorithm}.pkl"
            model_path = os.path.join('model', model_filename)
            joblib.dump(pipeline, model_path)
            print(f"      ✓ Model saved to {model_path}")
            
            # Store results
            results[algorithm] = {
                'accuracy': accuracy,
                'model_path': model_path,
                'classification_report': classification_report(y_test, y_pred)
            }
            
        except Exception as e:
            print(f"      ✗ Error training {algorithm}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        
        for algo, result in results.items():
            print(f"{algo.upper()}: {result['accuracy']*100:.2f}% accuracy")
        
        print(f"\nBEST MODEL: {best_model[0].upper()} ({best_model[1]['accuracy']*100:.2f}% accuracy)")
        
        # Show detailed report for best model
        print(f"\nDetailed Classification Report for {best_model[0].upper()}:")
        print(best_model[1]['classification_report'])
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("No models were successfully trained.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train intrusion detection models')
    parser.add_argument('--algorithms', nargs='+', 
                       choices=['randomforest', 'svm', 'xgboost'], 
                       default=['randomforest'],
                       help='Algorithms to train (default: randomforest)')
    
    args = parser.parse_args()
    
    train_intrusion_detection_model(args.algorithms)
