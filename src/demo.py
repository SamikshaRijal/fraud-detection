import sys
from pathlib import Path
import pickle
import pandas as pd
from preprocessing import get_data

models = [
    "Logistic Regression (Balanced)",
    "Logistic Regression (UnBalanced)",
    "Decision Tree (Balanced)",
    "Decision Tree (UnBalanced)",
    "Random Forest (Balanced)",
    "Random Forest (UnBalanced)",
    "Gradient Boosting (Balanced)",
    "Gradient Boosting (UnBalanced)",
    "K-Nearest Neighbors (KNN)"
]

# Display available models
for idx, name in enumerate(models, start=1):
    print(f"{idx}. {name}")

# Choose model
choice = int(input("Select a model (1-9): "))

if not 1 <= choice <= len(models):
    raise ValueError("Choose from 1-9")

# Get selected model name
name = models[choice - 1]

# Construct file path
fileName = f"results/models/{name.lower()}.pkl"

# Load saved model information
with open(fileName, "rb") as f:
    trained_model = pickle.load(f)

name = trained_model["name"]
model = trained_model["model"]
threshold = trained_model["threshold"]

# Get already-scaled test data
X_train, X_test, y_train, y_test = get_data()

# Select 2 normal and 2 fraudulent transactions
normal = X_test[y_test == 0].sample(2, random_state=42)
fraud = X_test[y_test == 1].sample(2, random_state=42)

samples = pd.concat([normal, fraud])

# Get actual labels for selected (samples) transactions
actual = y_test.loc[samples.index]

# Get fraud probabilities
probabilities = model.predict_proba(samples)[:, 1]

# Apply our tuned threshold
predictions = (probabilities >= threshold).astype(int)

# Print results
print(f"Model: {name}")
print(f"Threshold: {threshold:.4f}")

for idx, (actual, predicted, prob) in enumerate(zip(actual, predictions, probabilities), start=1):
    print(f"\nTransaction {idx}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print(f"Fraud probability: {prob:.4f}")

