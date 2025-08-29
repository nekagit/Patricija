import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the sample dataset
df = pd.read_csv('frontend/data/sample_credit_data.csv')

# Clean column names
df.columns = [col.strip() for col in df.columns]

# Separate features and target
target_col = 'loan_status'
feature_cols = [col for col in df.columns if col != target_col]

print(f"Features: {feature_cols}")
print(f"Target: {target_col}")
print(f"Dataset shape: {df.shape}")

# Prepare features and target
X = df[feature_cols]
y = df[target_col]

# Encode categorical features
categorical_features = []
numerical_features = []
encoders = {}

for col in feature_cols:
    if X[col].dtype == 'object' or pd.api.types.is_categorical_dtype(X[col].dtype):
        categorical_features.append(col)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"Encoded {col}: {dict(zip(le.classes_, range(len(le.classes_))))}")
    else:
        numerical_features.append(col)

print(f"Categorical features: {categorical_features}")
print(f"Numerical features: {numerical_features}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
scaler = StandardScaler()
if numerical_features:
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
else:
    X_train_scaled = X_train
    X_test_scaled = X_test

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model and artifacts
models_dir = Path("frontend/models")
models_dir.mkdir(exist_ok=True)

# Save model
joblib.dump(model, models_dir / "random_forest_model.pkl")

# Save scaler
joblib.dump(scaler, models_dir / "scaler.pkl")

# Save encoders
joblib.dump(encoders, models_dir / "label_encoders.pkl")

# Save feature names
with open(models_dir / "feature_names.json", "w") as f:
    json.dump(feature_cols, f, indent=2)

# LIME background data creation removed as requested

print(f"\nModel and artifacts saved to {models_dir}")
print(f"Feature names saved: {feature_cols}")
print(f"Model expects {len(feature_cols)} features")

# Test prediction with sample data
print("\nTesting prediction with first row of data:")
sample_data = X.iloc[0].to_dict()
print(f"Sample data: {sample_data}")

# Preprocess sample data
sample_df = pd.DataFrame([sample_data])
for col in categorical_features:
    if col in encoders:
        sample_df[col] = encoders[col].transform(sample_df[col].astype(str))

if numerical_features:
    sample_df[numerical_features] = scaler.transform(sample_df[numerical_features])

prediction = model.predict(sample_df)[0]
probability = model.predict_proba(sample_df)[0]
print(f"Prediction: {prediction}")
print(f"Probability: {probability}")
