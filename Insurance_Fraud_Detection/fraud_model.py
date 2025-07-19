"""
Insurance Claims Fraud Detection Model
1. Load and explore the claims dataset
2. Preprocess the data (handle missing values, outliers, feature scaling)
3. Train multiple models (at least 2 different algorithms)
4. Evaluate and compare using precision, recall, and F1-score
5. Select the best model and explain your choice

Success Criteria:
- Achieve precision > 0.75 (reduce false positives)
- Maintain recall > 0.60 (still catch fraud)
- Clear model comparison with business impact explanation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,  # ! TODO: use
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt


def load_and_explore_data():
    """
    Load the claims.csv dataset and perform EDA
    Returns:
        pd.DataFrame: The loaded claims dataset
    """
    df = pd.read_csv("data/claims.csv")
    # TODO: EDA
    return df


from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def preprocess_data(df):
    """
    Preprocess the data for machine learning

    Args:
        df (pd.DataFrame): Raw claims dataset

    Returns:
        tuple: (X_train, X_test, y_train, y_test) - preprocessed and split data
    """
    # Drop ID column
    df = df.drop(columns=["claim_id"])

    # Split features and target
    X = df.drop(columns="is_fraud")
    # ! TODO: add LLM processing of the description
    X = X.drop(columns=["description"])

    y = df["is_fraud"]

    # Column groups
    num_cols = ["claim_amount", "policy_age_days", "claimant_age", "previous_claims"]
    cat_cols = ["claim_type"]
    # text_col = "description"

    # Transformers
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # text_pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=100))])

    # Combine transformers
    preprocessor = ColumnTransformer(
        [
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
            # ("text", text_pipeline, text_col),
        ]
    )

    # Fit-transform features
    X_processed = preprocessor.fit_transform(X)

    # Train-test split
    return train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)


def train_models(X_train, y_train):
    """
    Train multiple machine learning models

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        dict: Dictionary of trained models
    """
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and compare their performance

    Args:
        models (dict): Dictionary of trained models
        X_test: Test features
        y_test: Test labels

    Returns:
        dict: Performance metrics for each model
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }

    # report = classification_report(y_test, y_pred, output_dict=True)
    # print(report)  # Access F1 for fraud class

    return results


def select_best_model(results):
    """
    Select the best model based on business requirements

    Args:
        results (dict): Performance metrics for each model

    Returns:
        str: Name of the best model with business justification
    """
    # Select model with highest F1-score
    best_model = max(results.items(), key=lambda x: x[1]["f1_score"])[0]

    print(
        f"Selected '{best_model}' as the best model based on highest F1-score, balancing precision and recall."
    )

    return best_model


def main():
    """
    Main function to run the complete fraud detection pipeline
    """
    print("=== Insurance Claims Fraud Detection ===\n")
    df = load_and_explore_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    best_model = select_best_model(results)
    print("Model Performance Comparison:")
    print(
        f"Random Forest: Precision={results['random_forest']['precision']}, Recall={results['random_forest']['recall']}, F1={results['logistic_regression']['f1_score']}"
    )
    print(
        f"Logistic Regression: Precision={results['logistic_regression']['precision']}, Recall={results['logistic_regression']['recall']}, F1={results['logistic_regression']['f1_score']}"
    )
    print()
    print(f"Recommended Model: {best_model}")
    print("Business Impact: [EXPLANATION]")
    X_train


if __name__ == "__main__":
    main()

# ! TODO: Business Context:
# TODO High claim amounts (>$10K) often require extra scrutiny
# TODO Customers with multiple previous claims have higher fraud risk
# TODO New policies (low policy_age_days) combined with high claims are suspicious
# * Balance is critical: false positives (FP) anger customers, missed fraud (FN) costs money
# FYI: recision = TP / (TP + FP), Recall = TP / (TP + FN)
