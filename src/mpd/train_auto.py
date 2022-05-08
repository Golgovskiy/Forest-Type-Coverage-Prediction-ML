from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .files import get_dataset


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
    "--k-folds-inner",
    default=3,
    type=int,
    show_default=True,
)
@click.option(
    "--k-folds-outer",
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
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--shuffle",
    default=True,
    type=bool,
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
    k_folds_inner: int,
    k_folds_outer: int,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    shuffle: bool,
) -> None:
    # create dataset
    features, target = get_dataset(
        dataset_path,
        random_state,
    )
    with mlflow.start_run():

        # configure the cross-validation procedure
        cv_inner = KFold(
            n_splits=k_folds_inner, shuffle=shuffle, random_state=random_state
        )

        # define search space
        space = dict()
        space["C"] = [1, 2, 5, 10]

        regressor = LogisticRegression(random_state=random_state, max_iter=max_iter)
        pipeline_steps = []
        if use_scaler:
            pipeline_steps.append(("scaler", StandardScaler()))

        pipeline_steps.append(
            (
                "search",
                GridSearchCV(
                    regressor,
                    space,
                    scoring="roc_auc_ovr",
                    n_jobs=-1,
                    cv=cv_inner,
                    refit=True,
                ),
            ),
        )
        pipeline = Pipeline(steps=pipeline_steps)

        # configure the cross-validation procedure
        cv_outer = KFold(n_splits=k_folds_outer, shuffle=True, random_state=1)
        # execute the nested cross-validation
        result = cross_validate(
            pipeline,
            features,
            target.values.ravel(),
            scoring=["accuracy", "roc_auc_ovr", "f1_weighted"],
            cv=cv_outer,
            n_jobs=-1,
        )

        fit_time = result["fit_time"]
        accuracy = result["test_accuracy"]
        f1 = result["test_f1_weighted"]
        roc_auc = result["test_roc_auc_ovr"]

        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("k-folds-inner", k_folds_inner)
        mlflow.log_param("k-folds-outer", k_folds_outer)

        mlflow.log_metric("accuracy", np.mean(accuracy))
        mlflow.log_metric("f1", np.mean(f1))
        mlflow.log_metric("roc_auc", np.mean(roc_auc))

        click.echo(
            f"Average fit time: {np.mean(fit_time):.3f} ± {np.std(fit_time):.3f}. Accuracy: {np.mean(accuracy):.3f} ± {np.std(accuracy):.3f}. F1: {np.mean(f1):.3f} ± {np.std(f1):.3f}. ROC_AUC: {np.mean(roc_auc):.3f} ± {np.std(roc_auc):.3f}. "
        )

        dump(pipeline, save_model_path)

        click.echo(f"Model is saved to {save_model_path}.")
