
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
df = pd.read_csv("dataset_01_customer_churn_risk.csv")

print(df.head())
print("Dataset shape:", df.shape)
print(df.dtypes)
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print(df["target"].value_counts())
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
StandardScaler()
model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic_regression", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_probability = model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)

print(cm)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
precision = precision_score(y_test, y_pred)

print("Precision:", precision)
recall = recall_score(y_test, y_pred)

print("Recall:", recall)
f1 = f1_score(y_test, y_pred)

print("F1 Score:", f1)
roc_auc = roc_auc_score(y_test, y_probability)

print("ROC-AUC:", roc_auc)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("Model Performance")
print("----------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)
print("ROC-AUC  :", roc_auc)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
coefficients = model.named_steps[
    "logistic_regression"
].coef_[0]

feature_names = X.columns

for feature, coefficient in zip(feature_names, coefficients):
    print(feature, ":", coefficient)