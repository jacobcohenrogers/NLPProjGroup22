# src/modeling.py
"""
Modeling module for authorship attribution pipeline.
Provides functions to load feature data, train classical ML models, evaluate them,
and save trained models. Designed to integrate with the existing notebook setup.

Usage in notebook:
    from modeling import load_data, train_models, evaluate_models, save_models
    X_train, X_test, y_train, y_test = load_data('data/features.csv')
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    save_models(models, 'models/')
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib


def load_data(features_csv: str = 'data/features.csv'):
    """
    Load feature matrix and split into train/test sets.
    Expects a CSV with columns: file_path, author, split, [feature columns...]
    Returns: X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(features_csv)
    # Ensure split column exists
    if 'split' not in df.columns:
        raise ValueError("Expected 'split' column in features CSV")
    # Train/test split based on 'split'
    train_df = df[df['split'] == 'training']
    test_df  = df[df['split'] == 'testing']

    y_train = train_df['author']
    y_test  = test_df['author']
    X_train = train_df.drop(columns=['file_path', 'author', 'prompt', 'split'])
    X_test  = test_df.drop(columns=['file_path', 'author', 'prompt', 'split'])

    return X_train, X_test, y_train, y_test


def train_models(X_train, y_train):
    """
    Train a set of classical models: Logistic Regression and Random Forest.
    Returns a dict of trained models.
    """
    models = {}
    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    models['random_forest'] = rf

    return models


def evaluate_models(models: dict, X_test, y_test):
    """
    Evaluate each model on test data. Returns a dict with metrics.
    Prints accuracy, precision, recall, F1, and confusion matrix.
    """
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'classes': model.classes_
        }
    return results


def save_models(models: dict, model_dir: str = 'models'):
    """
    Save trained models to disk using joblib. Each model saved as <model_dir>/<name>.joblib
    """
    os.makedirs(model_dir, exist_ok=True)
    for name, model in models.items():
        path = Path(model_dir) / f"{name}.joblib"
        joblib.dump(model, path)
    return {name: str(Path(model_dir) / f"{name}.joblib") for name in models}


if __name__ == '__main__':
    # CLI: load features, train, evaluate, and save
    X_train, X_test, y_train, y_test = load_data()
    models = train_models(X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    for name, metrics in results.items():
        print(f"Model: {name}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
        print(f"  Confusion matrix (rows=actual, cols=predicted):")
        print(metrics['confusion_matrix'])
        print()
    saved = save_models(models)
    print(f"Saved models: {saved}")