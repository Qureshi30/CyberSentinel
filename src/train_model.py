"""
Model Training Script
Trains a RandomForest classifier on NSL-KDD dataset
"""

import os
import sys
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Add parent directory to path to import preprocess module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_csv, clean_data, encode_categorical_features, convert_labels_to_binary


def train_intrusion_detection_model():
    """
    Main function to train the intrusion detection model
    """
    print("=" * 60)
    print("INTRUSION DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Define file paths
    train_data_path = os.path.join('data', 'train', 'KDDTrain+.csv')
    model_output_path = os.path.join('model', 'intrusion_model.pkl')
    encoders_output_path = os.path.join('model', 'label_encoders.pkl')
    
    # Check if training data exists
    if not os.path.exists(train_data_path):
        print(f"ERROR: Training data not found at {train_data_path}")
        print("Please place KDDTrain+.csv in the data/train/ directory")
        return
    
    # Step 1: Load the training data
    print(f"\n[1/6] Loading training data from {train_data_path}...")
    df = load_csv(train_data_path)
    if df is None:
        return
    print(f"   ✓ Loaded {len(df)} records with {len(df.columns)} columns")
    
    # Step 2: Clean the data
    print("\n[2/6] Cleaning data...")
    df = clean_data(df)
    print(f"   ✓ Data cleaned. {len(df)} records remaining")
    
    # Step 3: Separate features and labels
    print("\n[3/6] Preparing features and labels...")
    # Assume the last column is the label
    if 'label' in df.columns:
        label_column = 'label'
    elif 'class' in df.columns:
        label_column = 'class'
    else:
        # Use the last column as label
        label_column = df.columns[-1]
    
    # Convert labels to binary (Threat/Harmless)
    y = convert_labels_to_binary(df[label_column])
    X = df.drop(columns=[label_column])
    
    print(f"   ✓ Features: {X.shape[1]} columns")
    print(f"   ✓ Labels: {len(y)} samples")
    print(f"   ✓ Class distribution:")
    print(f"      - Harmless: {(y == 'Harmless').sum()}")
    print(f"      - Threat: {(y == 'Threat').sum()}")
    
    # Step 4: Encode categorical features
    print("\n[4/6] Encoding categorical features...")
    X, label_encoders = encode_categorical_features(X)
    print(f"   ✓ Encoded {len(label_encoders)} categorical columns")
    
    # Step 5: Split data and train the model
    print("\n[5/6] Training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train RandomForest classifier
    model = RandomForestClassifier(
        n_estimators=100,  # number of trees
        random_state=42,
        n_jobs=-1  # use all CPU cores
    )
    model.fit(X_train, y_train)
    print("   ✓ Model training complete")
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n   Model Accuracy: {accuracy * 100:.2f}%")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 6: Save the trained model and encoders
    print("\n[6/6] Saving model and encoders...")
    os.makedirs('model', exist_ok=True)
    joblib.dump(model, model_output_path)
    joblib.dump(label_encoders, encoders_output_path)
    print(f"   ✓ Model saved to {model_output_path}")
    print(f"   ✓ Encoders saved to {encoders_output_path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    train_intrusion_detection_model()
