import mlflow


def log_mlflow_data(estimator_params: dict, metrics: dict) -> None:  # type: ignore
    mlflow.log_params(estimator_params)
    mlflow.log_metrics(metrics)
