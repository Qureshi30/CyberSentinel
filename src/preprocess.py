import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def clean_data(df):
    df = df.drop_duplicates()
    
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    return df


def encode_categorical_features(df, label_encoders=None):
    if label_encoders is None:
        label_encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    if 'label' in categorical_columns:
        categorical_columns = categorical_columns.drop('label')
    
    for col in categorical_columns:
        if col not in label_encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col].astype(str))
    
    return df, label_encoders


def convert_labels_to_binary(labels):
    return labels.apply(
        lambda x: 'Harmless' if str(x).strip().lower() == 'normal' else 'Threat'
    )


def balance_dataset(df, label_column='label', random_state=42):
    """
    Returns a roughly balanced dataset by undersampling the majority class.

    Works on binary intent (Harmless vs Threat) even when the label column
    contains raw NSL-KDD attack names.

    Keeps all minority-class samples and randomly samples the majority class
    more aggressively to bias training toward recall.
    """
    if label_column not in df.columns:
        raise KeyError(f"Label column '{label_column}' not found in dataframe")

    balanced_df = df.copy()

    def _to_binary_label(value):
        value_str = str(value).strip().lower()
        harmless_values = {"normal", "harmless", "benign", "0"}
        return "Harmless" if value_str in harmless_values else "Threat"

    balanced_df["_binary_label"] = balanced_df[label_column].apply(_to_binary_label)
    class_counts = balanced_df["_binary_label"].value_counts()

    if class_counts.empty:
        raise ValueError("Cannot balance an empty dataframe")

    if len(class_counts) < 2:
        raise ValueError("Dataset must contain at least two classes to balance")

    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    minority_count = int(class_counts[minority_class])
    majority_count = int(class_counts[majority_class])

    if minority_count == 0:
        raise ValueError("Cannot balance a dataset when one class has zero samples")

    # Aggressively undersample the majority class while keeping all minority samples.
    # This intentionally makes the training set harder and reduces majority bias.
    target_majority_size = max(1, int(round(minority_count * 0.8)))
    target_majority_size = min(majority_count, target_majority_size)

    minority_df = balanced_df[balanced_df["_binary_label"] == minority_class]
    majority_df = balanced_df[balanced_df["_binary_label"] == majority_class]

    sampled_majority_df = majority_df.sample(n=target_majority_size, random_state=random_state)

    balanced_dataset = pd.concat([minority_df, sampled_majority_df], axis=0)
    balanced_dataset = balanced_dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)

    balanced_dataset = balanced_dataset.drop(columns=["_binary_label"])

    return balanced_dataset


# ============================
# 🔍 NEW: ANALYSIS FUNCTIONS
# ============================

def analyze_class_distribution(df, label_column='label'):
    """
    Analyze original multi-class distribution
    """
    counts = df[label_column].value_counts()
    total = len(df)

    print("\n📊 Multi-class Distribution:")
    for label, count in counts.items():
        print(f"{label}: {count} ({(count/total)*100:.2f}%)")

    return counts


def analyze_binary_skew(labels, threshold=1.5):
    """
    Analyze skew after converting to binary classes
    
    Args:
        labels: Series (already binary: Harmless / Threat)
        threshold: imbalance threshold ratio
    """
    counts = labels.value_counts()
    total = len(labels)

    harmless = counts.get('Harmless', 0)
    threat = counts.get('Threat', 0)

    print("\n🛡️ Binary Class Distribution:")
    print(f"Harmless: {harmless} ({(harmless/total)*100:.2f}%)")
    print(f"Threat: {threat} ({(threat/total)*100:.2f}%)")

    if harmless == 0 or threat == 0:
        print("⚠️ One class missing — invalid dataset for classification")
        return

    ratio = max(harmless, threat) / min(harmless, threat)
    print(f"\nImbalance Ratio: {ratio:.2f}")

    if ratio > threshold:
        dominant = "Threat" if threat > harmless else "Harmless"
        print(f"⚠️ Dataset is skewed towards '{dominant}'")
    else:
        print("✅ Dataset is fairly balanced")

    return {
        "harmless": harmless,
        "threat": threat,
        "ratio": ratio
    }