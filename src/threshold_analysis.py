
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

df = pd.read_csv("data/telco_churn_cleaned.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_inner, X_val, y_train_inner, y_val = train_test_split(
    X_train, y_train, test_size=0.25,
    random_state=42, stratify=y_train
)

model = joblib.load("models/logistic_regression_pipeline.joblib")

print("Threshold Comparison")
print("-" * 55)
print("Threshold | Precision | Recall | F1 Score")

# This saved model was fitted on X_train, including X_val.
# Refit a fresh copy only on X_train_inner for valid threshold selection.
from sklearn.base import clone

validation_model = clone(model)
validation_model.fit(X_train_inner, y_train_inner)

probabilities = validation_model.predict_proba(X_val)[:, 1]

results = []

for threshold in [0.30, 0.40, 0.50, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(y_val, predictions, zero_division=0)
    recall = recall_score(y_val, predictions, zero_division=0)
    f1 = f1_score(y_val, predictions, zero_division=0)

    results.append([threshold, precision, recall, f1])

    print(f"{threshold:.2f}      | {precision:.4f}    | {recall:.4f} | {f1:.4f}")

best = max(results, key=lambda row: row[3])

print("\nBest Validation Threshold:", best[0])
print("Best Validation F1 Score:", round(best[3], 4))


print("\nFinal Test Set Evaluation")
print("-" * 45)

test_probabilities = model.predict_proba(X_test)[:, 1]

for threshold in [0.30, 0.50]:
    predictions = (test_probabilities >= threshold).astype(int)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print(f"\nThreshold: {threshold}")
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
