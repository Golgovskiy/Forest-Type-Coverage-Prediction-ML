import mlflow
import click


def log_mlflow_data(estimator_params: dict, metrics: dict) -> None:
    mlflow.log_params(estimator_params)
    mlflow.log_metrics(metrics)
