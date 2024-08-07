from pathlib import Path

import click
import mlflow

from mpd import files, model, crossvalidate, mlflow_utils


@click.command()
@click.option(
    "--c",
    default=1,
    help="Inverse of regularization strength;"
    " must be a positive float. Like in support vector "
    "machines, smaller values specify stronger regularization.",
    type=float,
    show_default=True,
)
@click.option(
    "--penalty",
    default="l2",
    help="Specify the norm of the penalty: 'l1'," " 'l2', 'elasticnet', 'none'.",
    type=click.Choice(["l1", "l2", "elasticnet", "none"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--shuffle",
    default=True,
    help="Shuffle features before splitting.",
    type=bool,
    show_default=True,
)
@click.option(
    "-d",
    "--dataset_path",
    default="data/train.csv",
    help="Path to dataset.",
    show_default=True,
)
@click.option(
    "-f",
    "--only_fit",
    default=False,
    type=bool,
    help="Only fit classificator and save it.",
    show_default=True,
)
@click.option(
    "-s",
    "--save_model",
    help="Whether to save fitted model or not.",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--save_model_path",
    help="Path to save fitted model to.",
    default="data/logreg.joblib",
    type=Path,
    show_default=True,
)
@click.option(
    "--random_state", default=42, help="Random seed.", type=int, show_default=True
)
@click.option(
    "--cv_splits",
    default=20,
    help="Specify the number of folds in cross-validation splitting.",
    type=int,
    show_default=True,
)
@click.option(
    "--use_scaler",
    default=True,
    help="Use StandardScaler to scale dataset.",
    type=bool,
    show_default=True,
)
@click.option(
    "--max_iter",
    default=2000,
    help="Maximum number of iterations.",
    type=int,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    cv_splits: int,
    use_scaler: bool,
    max_iter: int,
    c: float,
    penalty: str,
    save_model: bool,
    only_fit: bool,
    shuffle: bool,
) -> None:

    mlflow.set_experiment("logreg")
    (features, target) = files.get_dataset(dataset_path)
    estimator = model.get_logreg(
        max_iter=max_iter, c=c, penalty=penalty, random_state=random_state
    )
    pipe = model.create_pipeline(estimator, use_scaler)
    if not only_fit:
        with mlflow.start_run():
            metrics = crossvalidate.cross_validate_model(
                features=features,
                target=target,
                cv_splits=cv_splits,
                shuffle=shuffle,
                use_scaler=use_scaler,
                random_state=random_state,
                estimator=pipe,
            )
            mlflow_utils.log_mlflow_data(
                {"C": c, "penalty": penalty, "max_iter": max_iter}, metrics
            )

    if save_model:
        click.echo("Fitting model...")
        pipe.fit(features, target.values.ravel())
        files.save_the_model(pipe, use_scaler, save_model_path)
    click.echo("Finished.")
