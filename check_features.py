import pandas as pd
import json

# Check what features the model expects
with open('frontend/models/feature_names.json', 'r') as f:
    model_features = json.load(f)

print("Model expects these features:")
for i, feature in enumerate(model_features):
    print(f"{i+1}. {feature}")

print(f"\nTotal features model expects: {len(model_features)}")

# Check what features the sample dataset has
df = pd.read_csv('frontend/data/sample_credit_data.csv')
sample_features = [col.strip() for col in df.columns if col.strip() != 'loan_status']

print("\nSample dataset has these features:")
for i, feature in enumerate(sample_features):
    print(f"{i+1}. {feature}")

print(f"\nTotal features in sample dataset: {len(sample_features)}")

# Check overlap
overlap = set(model_features) & set(sample_features)
print(f"\nOverlapping features: {len(overlap)}")
for feature in overlap:
    print(f"- {feature}")

# Check missing features
missing_in_model = set(sample_features) - set(model_features)
print(f"\nFeatures in sample but not in model: {len(missing_in_model)}")
for feature in missing_in_model:
    print(f"- {feature}")

missing_in_sample = set(model_features) - set(sample_features)
print(f"\nFeatures in model but not in sample: {len(missing_in_sample)}")
for feature in missing_in_sample:
    print(f"- {feature}")
