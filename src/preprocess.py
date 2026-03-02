"""
Preprocessing module for network traffic data
This module contains shared functions for cleaning and preparing data
Note: Feature encoding is now handled by sklearn pipelines in the model files
"""

import pandas as pd


def load_csv(file_path):
    """
    Load CSV file into a pandas DataFrame
    
    Args:
        file_path: path to the CSV file
    
    Returns:
        DataFrame containing the CSV data
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def clean_data(df):
    """
    Clean the dataset by handling missing values and removing duplicates
    
    Args:
        df: input DataFrame
    
    Returns:
        cleaned DataFrame
    """
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Fill missing numeric values with median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill missing categorical values with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def encode_categorical_features(df, label_encoders=None):
    """
    Convert categorical text features to numeric values
    
    Args:
        df: input DataFrame
        label_encoders: dictionary of pre-fitted LabelEncoders (for prediction)
    
    Returns:
        DataFrame with encoded features, dictionary of label encoders
    """
    if label_encoders is None:
        label_encoders = {}
    
    # Identify categorical columns (excluding the label column if present)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if 'label' in categorical_columns:
        categorical_columns = categorical_columns.drop('label')
    
    # Encode each categorical column
    for col in categorical_columns:
        if col not in label_encoders:
            # Create new encoder for training
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            # Use existing encoder for prediction
            le = label_encoders[col]
            # Handle unseen labels by assigning them to a default value
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    
    return df, label_encoders


def convert_labels_to_binary(labels):
    """
    Convert attack labels to binary classification:
    - 'normal' -> 'Harmless'
    - everything else -> 'Threat'
    
    Args:
        labels: pandas Series of original labels
    
    Returns:
        pandas Series of binary labels
    """
    binary_labels = labels.apply(lambda x: 'Harmless' if str(x).strip().lower() == 'normal' else 'Threat')
    return binary_labels

