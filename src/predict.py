"""
Prediction Script
Loads trained models and makes predictions on new network traffic data
Supports: RandomForest, SVM, XGBoost
"""

import os
import sys
import pandas as pd
import joblib

# Add parent directory to path to import preprocess module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_csv, clean_data

# Test data file paths
TEST_DATA_PATHS = {
    'test21': os.path.join('data', 'test', 'KDDTest-21.csv'),
    'testplus': os.path.join('data', 'test', 'KDDTest+.csv')
}


def load_model(algorithm='randomforest'):
    """
    Load the trained model for specified algorithm
    
    Args:
        algorithm: 'randomforest', 'svm', or 'xgboost'
        
    Returns:
        model: trained ML model pipeline
    """
    model_path = os.path.join('model', f'{algorithm}.pkl')
    
    if not os.path.exists(model_path):
        available_models = []
        for algo in ['randomforest', 'svm', 'xgboost']:
            if os.path.exists(os.path.join('model', f'{algo}.pkl')):
                available_models.append(algo)
        
        if available_models:
            raise FileNotFoundError(f"Model '{algorithm}' not found at {model_path}. Available models: {available_models}")
        else:
            raise FileNotFoundError(f"No trained models found. Please train models first using train_model.py")
    
    model = joblib.load(model_path)
    return model


def list_available_models():
    """
    List all available trained models
    
    Returns:
        list of available algorithm names
    """
    available = []
    for algo in ['randomforest', 'svm', 'xgboost']:
        if os.path.exists(os.path.join('model', f'{algo}.pkl')):
            available.append(algo)
    return available


def predict_from_csv(csv_file_path, algorithm='randomforest'):
    """
    Make predictions on network traffic data from a CSV file
    
    Args:
        csv_file_path: path to the CSV file containing network traffic data
        algorithm: algorithm to use ('randomforest', 'svm', 'xgboost')
    
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
    
    # Drop difficulty column if exists
    if "difficulty" in df_clean.columns:
        df_clean = df_clean.drop("difficulty", axis=1)
    
    # Load the trained model
    print(f"Loading {algorithm.upper()} model...")
    model = load_model(algorithm)
    
    # Remove label column if it exists (for testing with labeled data)
    label_columns = ['label', 'class']
    actual_labels = None
    for label_col in label_columns:
        if label_col in df_clean.columns:
            actual_labels = df_clean[label_col].copy()
            df_clean = df_clean.drop(columns=[label_col])
            break
    
    # Make predictions
    print("Making predictions...")
    try:
        predictions = model.predict(df_clean)
        print(f"   ✓ Predictions completed for {len(predictions)} records")
        
        # Convert predictions to labels
        prediction_labels = ['normal' if pred == 0 else 'attack' for pred in predictions]
        
        # Add predictions to dataframe
        df['prediction'] = prediction_labels
        df['prediction_numeric'] = predictions
        df['model_used'] = algorithm
        
        # Add original labels if they existed
        if actual_labels is not None:
            df['actual_label'] = actual_labels
            
            # Calculate accuracy if we have original labels
            actual_binary = [0 if str(label).strip().lower() == 'normal' else 1 for label in actual_labels]
            accuracy = sum(pred == actual for pred, actual in zip(predictions, actual_binary)) / len(predictions)
            print(f"   ✓ Accuracy: {accuracy * 100:.2f}%")
            df['correct_prediction'] = [pred == actual for pred, actual in zip(predictions, actual_binary)]
        
        # Calculate statistics
        threat_count = sum(predictions)
        harmless_count = len(predictions) - threat_count
        
        print(f"\nPrediction Summary:")
        print(f"  Total: {len(predictions)}")
        print(f"  Attack: {threat_count} ({threat_count/len(predictions)*100:.1f}%)")
        print(f"  Normal: {harmless_count} ({harmless_count/len(predictions)*100:.1f}%)")
        
        return df
        
    except Exception as e:
        print(f"Error making predictions: {str(e)}")
        return None


def predict_batch(data_dict_list, algorithm='randomforest'):
    """
    Make predictions on a list of data dictionaries (used by Flask app)
    
    Args:
        data_dict_list: list of dictionaries containing feature values
        algorithm: algorithm to use ('randomforest', 'svm', 'xgboost')
    
    Returns:
        list of predictions
    """
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data_dict_list)
    
    # Clean the data
    df_clean = clean_data(df.copy())
    
    # Drop difficulty column if exists
    if "difficulty" in df_clean.columns:
        df_clean = df_clean.drop("difficulty", axis=1)
    
    # Load model
    model = load_model(algorithm)
    
    # Remove label column if it exists
    label_columns = ['label', 'class']
    for label_col in label_columns:
        if label_col in df_clean.columns:
            df_clean = df_clean.drop(columns=[label_col])
            break
    
    # Make predictions
    predictions = model.predict(df_clean)
    
    # Convert to readable labels
    prediction_labels = ['normal' if pred == 0 else 'attack' for pred in predictions]
    
    return prediction_labels


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using trained models')
    parser.add_argument('input_csv', nargs='?', help='Path to the CSV file for prediction')
    parser.add_argument('--algorithm', 
                       choices=['randomforest', 'svm', 'xgboost'], 
                       default='randomforest',
                       help='Algorithm to use for prediction (default: randomforest)')
    parser.add_argument('--test', 
                       choices=['test21', 'testplus'], 
                       help='Use predefined test dataset (test21=KDDTest-21.csv, testplus=KDDTest+.csv)')
    parser.add_argument('--list-models', action='store_true', 
                       help='List available trained models')
    
    args = parser.parse_args()
    
    if args.list_models:
        available = list_available_models()
        if available:
            print(f"Available trained models: {', '.join(available)}")
        else:
            print("No trained models found. Please train models first using train_model.py")
        sys.exit(0)
    
    # Determine input file
    if args.test:
        input_file = TEST_DATA_PATHS[args.test]
        print(f"Using test dataset: {input_file}")
    elif args.input_csv:
        input_file = args.input_csv
    else:
        print("ERROR: Please provide either --test or input_csv path")
        print("Available test datasets:")
        for key, path in TEST_DATA_PATHS.items():
            print(f"  --test {key}: {path}")
        sys.exit(1)
    
    if not os.path.exists(input_file):
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)
    
    # Check if the selected model is available
    available_models = list_available_models()
    if args.algorithm not in available_models:
        print(f"ERROR: Model '{args.algorithm}' not trained. Available models: {available_models}")
        print("Train models using: python train_model.py --algorithms randomforest svm xgboost")
        sys.exit(1)
    
    try:
        # Make predictions
        result_df = predict_from_csv(input_file, args.algorithm)
        
        if result_df is not None:
            # Save results
            output_path = input_file.replace('.csv', f'_predictions_{args.algorithm}.csv')
            result_df.to_csv(output_path, index=False)
            print(f"\nPredictions saved to: {output_path}")
        else:
            print("Failed to make predictions")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
