"""
Prediction Script
Loads trained model and makes predictions on new network traffic data
"""

import os
import sys
import pandas as pd
import joblib

# Add parent directory to path to import preprocess module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_csv, clean_data, encode_categorical_features


def load_model():
    """
    Load the trained model and label encoders from disk
    
    Returns:
        model: trained ML model
        label_encoders: dictionary of label encoders
    """
    model_path = os.path.join('model', 'intrusion_model.pkl')
    encoders_path = os.path.join('model', 'label_encoders.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    if not os.path.exists(encoders_path):
        raise FileNotFoundError(f"Encoders not found at {encoders_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    
    return model, label_encoders


def predict_from_csv(csv_file_path):
    """
    Make predictions on network traffic data from a CSV file
    
    Args:
        csv_file_path: path to the CSV file containing network traffic data
    
    Returns:
        pandas DataFrame with original data and predictions
    """
    print(f"Loading data from {csv_file_path}...")
    
    # Load the CSV file
    df = load_csv(csv_file_path)
    if df is None:
        raise ValueError("Failed to load CSV file")
    
    print(f"Loaded {len(df)} records")
    
    # Clean the data
    df_clean = clean_data(df.copy())
    
    # Load the trained model and encoders
    print("Loading trained model...")
    model, label_encoders = load_model()
    
    # Remove label column if it exists (for testing with labeled data)
    label_columns = ['label', 'class']
    actual_labels = None
    for label_col in label_columns:
        if label_col in df_clean.columns:
            actual_labels = df_clean[label_col].copy()
            df_clean = df_clean.drop(columns=[label_col])
            break
    
    # Encode categorical features using the same encoders from training
    print("Preprocessing data...")
    df_encoded, _ = encode_categorical_features(df_clean, label_encoders)
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(df_encoded)
    
    # Add predictions to the original dataframe
    df['Prediction'] = predictions
    
    # Calculate statistics
    threat_count = (predictions == 'Threat').sum()
    harmless_count = (predictions == 'Harmless').sum()
    
    print(f"\nPrediction Summary:")
    print(f"  Total: {len(predictions)}")
    print(f"  Threat: {threat_count} ({threat_count/len(predictions)*100:.1f}%)")
    print(f"  Harmless: {harmless_count} ({harmless_count/len(predictions)*100:.1f}%)")
    
    return df


def predict_batch(data_dict_list):
    """
    Make predictions on a list of data dictionaries (used by Flask app)
    
    Args:
        data_dict_list: list of dictionaries containing feature values
    
    Returns:
        list of predictions
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_dict_list)
    
    # Clean the data
    df_clean = clean_data(df.copy())
    
    # Load model and encoders
    model, label_encoders = load_model()
    
    # Remove label column if it exists
    label_columns = ['label', 'class']
    for label_col in label_columns:
        if label_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[label_col])
            break
    
    # Encode categorical features
    df_encoded, _ = encode_categorical_features(df_clean, label_encoders)
    
    # Make predictions
    predictions = model.predict(df_encoded)
    
    return predictions.tolist()


if __name__ == "__main__":
    # Command-line usage
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_csv>")
        print("Example: python predict.py data/test/sample_input.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    
    if not os.path.exists(input_csv):
        print(f"ERROR: File not found: {input_csv}")
        sys.exit(1)
    
    # Make predictions
    result_df = predict_from_csv(input_csv)
    
    # Save results
    output_path = input_csv.replace('.csv', '_predictions.csv')
    result_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
