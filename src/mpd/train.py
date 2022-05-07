from pathlib import Path
from joblib import dump

import click
# import mlflow
# import mlflow.sklearn
import sklearn
from sklearn.metrics import accuracy_score,roc_auc_score,f1_score
from sklearn.model_selection import cross_validate

from .data import get_dataset
from .pipeline import create_pipeline


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--k-folds",
    default=5,
    type=int,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    k_folds: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
) -> None:
    features, target = get_dataset(
        dataset_path,
        random_state,
        k_folds,
    )
    
    if(k_folds <= 1 or k_folds >=len(features):
        raise click.ClickException("Wrong k-folds value!")
        
    with mlflow.start_run():
    
        pipeline = create_pipeline(use_scaler, max_iter, logreg_c, random_state)
        
        result = cross_validate(pipeline, features, target, scoring = ['f1','accuracy', 'roc_auc'], cv=k_folds)
        
        accuracy = np.mean(result["test_score"]["accuracy"])
        f1 = np.mean(result["test_score"]["f1"])
        roc_auc = np.mean(result["test_score"]["roc_auc"])
        
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("k-folds", k-folds)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        click.echo(f"Accuracy: {accuracy}. F1: {f1}. ROC_AUC: {roc_auc}.")
        
        dump(pipeline, save_model_path)
        
        click.echo(f"Model is saved to {save_model_path}.")