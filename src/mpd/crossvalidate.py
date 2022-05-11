import numpy as np
from pandas import DataFrame, Series
import click
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold

from mpd import model


def cross_validate_model(
    features: DataFrame,
    target: Series,
    cv_splits: int,
    shuffle: bool,
    estimator: BaseEstimator,
    random_state: int,
    use_scaler: bool,
) -> dict:  # type: ignore
    click.echo("Cross-validating model...")

    pipe = model.create_pipeline(estimator, use_scaler)

    cv_outer = StratifiedKFold(
        n_splits=cv_splits, shuffle=shuffle, random_state=random_state
    )
    result = cross_validate(
        pipe,
        features,
        target.values.ravel(),
        scoring=["accuracy", "roc_auc_ovr", "f1_weighted"],
        cv=cv_outer,
        n_jobs=-1,
        return_estimator=True,
    )
    metrics = {
        "test_accuracy": float(np.mean(result["test_accuracy"])),
        "test_f1_weighted": float(np.mean(result["test_f1_weighted"])),
        "test_roc_auc_ovr": float(np.mean(result["test_roc_auc_ovr"])),
        "fit_time": float(np.mean(result["fit_time"])),
    }

    click.echo(
        f"Average fit time: {metrics['fit_time']:.3f} ±"
        f" {np.std(metrics['fit_time']):.3f}. "
        f"Accuracy: {np.mean(metrics['test_accuracy']):.3f} ±"
        f" {np.std(metrics['test_accuracy']):.3f}. "
        f"F1: {np.mean(metrics['test_f1_weighted']):.3f} ±"
        f" {np.std(metrics['test_f1_weighted']):.3f}. "
        f"ROC_AUC: {np.mean(metrics['test_roc_auc_ovr']):.3f} ±"
        f" {np.std(metrics['test_roc_auc_ovr']):.3f}."
    )
    return metrics
