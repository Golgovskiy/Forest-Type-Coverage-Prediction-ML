from pathlib import Path

import click
import mlflow

from mpd import files
from mpd import model
from mpd import searchcv
from mpd import crossvalidate
from mpd import mlflow_utils


@click.command()
@click.option(
    "-m",
    "--model_type",
    default="logreg",
    type=click.Choice(["logreg", "randfor"], case_sensitive=False),
    help="Model to train",
    show_default=True,
)
@click.option(
    "-d",
    "--dataset_path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to dataset.",
    show_default=True,
)
@click.option(
    "-st",
    "--save_model_path",
    help="Path to save fitted model to.",
    default="data/logreg.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
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
    "--search_cv",
    default="random",
    type=click.Choice(["random", "grid"], case_sensitive=False),
    help="Hyperparameter estimator strategy",
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
    "--random_state",
    default=42,
    help="Random seed.",
    type=int,
    show_default=True,
)
@click.option(
    "--cv_splits_inner",
    default=5,
    help="Number of splits in inner loop.",
    type=int,
    show_default=True,
)
@click.option(
    "--cv_splits_outer",
    default=5,
    help="Number of splits in outer loop.",
    type=int,
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
    "--use_scaler",
    default=True,
    help="Use StandardScaler to scale dataset.",
    type=bool,
    show_default=True,
)
def train(
    model_type: str,
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    cv_splits_inner: int,
    cv_splits_outer: int,
    shuffle: bool,
    search_cv: str,
    use_scaler: bool,
    save_model: bool,
    only_fit: bool,
) -> None:
    features, target = files.get_dataset(dataset_path)
    click.echo("Fitting model...")
    searcher = searchcv.get_search(
        search_type=search_cv,
        random_state=random_state,
        cv_splits=cv_splits_inner,
        shuffle=shuffle,
        model_type=model.model_types[model_type],
    )
    pipe = model.create_pipeline(searcher, use_scaler)
    estimator, params = searchcv.search(
        searcher=pipe, features=features, targets=target, use_scaler=use_scaler
    )
    pipe2 = model.create_pipeline(estimator, use_scaler)
    if not only_fit:
        mlflow.set_experiment(f"{model_type}_auto")
        with mlflow.start_run():
            metrics = crossvalidate.cross_validate_model(
                features=features,
                target=target,
                cv_splits=cv_splits_outer,
                shuffle=shuffle,
                use_scaler=use_scaler,
                random_state=random_state,
                estimator=pipe2,
            )
            mlflow_utils.log_mlflow_data(params, metrics)

    fit_pipe = pipe2.fit(features, target.values.ravel())
    if save_model:
        files.save_the_model(fit_pipe, use_scaler, save_model_path)

    click.echo("Finished.")
