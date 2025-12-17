
import pickle
import numpy as np

with open('reports/models/logreg_model.pkl', 'rb') as f:
    model = pickle.load(f)

print(f"Model trained on: {model.n_features_in_} features")

# Load data
import pandas as pd
df = pd.read_csv('data/features/features_engineered_monthly.csv')
X_data = df.drop(['date', 'recession_label'], axis=1)
y_data = df['recession_label']

print(f"Data has: {X_data.shape[1]} features")
print(f"Match: {X_data.shape[1] == model.n_features_in_}")

# Try a prediction
try:
    pred = model.predict(X_data.iloc[150:151].values)
    print(f"Prediction works: {pred[0]}")
    proba = model.predict_proba(X_data.iloc[150:151].values)
    print(f"Probabilities: {proba[0]}")
except Exception as e:
    print(f"Error: {e}")




