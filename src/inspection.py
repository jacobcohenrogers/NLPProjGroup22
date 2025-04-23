# src/inspection.py
"""
Model inspection and evaluation utilities for the authorship attribution pipeline.
Provides functions to load saved models, summarize their parameters,
compute classification reports, confusion matrices, and feature importances.

Usage in notebook:
    from inspection import (
        load_models,
        classification_report_df,
        confusion_matrix_df,
        top_feature_importances
    )
    models = load_models('models/')
    report = classification_report_df(models['random_forest'], X_test, y_test)
    cm_df = confusion_matrix_df(models['random_forest'], X_test, y_test)
    fi = top_feature_importances(models['random_forest'], feature_names)
"""
import os
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def load_models(model_dir: str) -> dict:
    """
    Load all .joblib models from a directory.
    Returns a dict mapping model name (filename without extension) to estimator.
    """
    models = {}
    model_dir = Path(model_dir)
    for file in model_dir.glob('*.joblib'):
        name = file.stem
        models[name] = joblib.load(file)
    return models


def classification_report_df(model, X, y_true) -> pd.DataFrame:
    """
    Generate a classification report as a DataFrame (precision, recall, f1, support).
    """
    y_pred = model.predict(X)
    report_dict = classification_report(y_true, y_pred, output_dict=True)
    df = pd.DataFrame(report_dict).transpose()
    return df


def confusion_matrix_df(model, X, y_true) -> pd.DataFrame:
    """
    Compute confusion matrix and return as a labeled DataFrame.
    Rows are true labels, columns are predicted labels.
    """
    y_pred = model.predict(X)
    labels = model.classes_
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    return df_cm


def top_feature_importances(model, feature_names, top_n: int = 10) -> pd.DataFrame:
    """
    For tree-based or linear models, return a DataFrame of top_n features by importance or coefficient.
    Works with models having `feature_importances_` or `coef_` attribute.
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # for multiclass, sum absolute coefficients
        coefs = model.coef_
        importances = abs(coefs).sum(axis=0)
    else:
        raise ValueError("Model does not have feature_importances_ or coef_ attribute.")

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df_sorted = df.sort_values('importance', ascending=False).head(top_n)
    return df_sorted.reset_index(drop=True)
