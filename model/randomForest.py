import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("data/train/KDDTrain+.csv")

# Drop difficulty column if exists
if "difficulty" in df.columns:
    df = df.drop("difficulty", axis=1)

# Convert label to binary
df["label"] = df["label"].apply(lambda x: 0 if x == "normal" else 1)

X = df.drop("label", axis=1)
y = df["label"]

# Categorical columns
categorical_cols = ["protocol_type", "service", "flag"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# Model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Random Forest Results:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, "model/random_forest.pkl")
print("Model saved to model/random_forest.pkl")
