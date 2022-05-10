from pathlib import Path

import click
import mlflow.sklearn as mlflow

from mpd import files, model, crossvalidate, mlflow_utils


@click.command()
@click.option(
    "--n_estimators",
    default=100,
    help="The number of trees in the forest.",
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    help="The function to measure the quality of a split. Supported criteria are"
    "  'gini' for the Gini impurity and 'entropy' for the information gain."
    " Note: this parameter is tree-specific.",
    type=click.Choice(["gini", "entropy"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--max_depth",
    default=None,
    help="The maximum depth of the tree. If None, then nodes are expanded until all "
    "leaves are pure or until all leaves contain less than min_samples_split samples.",
    type=int or None,
    show_default=True,
)
@click.option(
    "--min_samples_split",
    default=2,
    help="The minimum number of samples required to split an internal node.",
    type=int,
    show_default=True,
)
@click.option(
    "--bootstrap",
    default=True,
    help="Whether bootstrap samples are used when building trees. If False, "
    "the whole dataset is used to build each tree.",
    type=bool,
    show_default=True,
)
@click.option(
    "--ccp_alpha",
    default=0,
    help="(RandomForestClassifier) Complexity parameter used for Minimal Cost-Complexity Pruning. "
    "The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen."
    " By default, no pruning is performed.",
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
def train(
    dataset_path,
    save_model_path: Path,
    random_state: Path,
    use_scaler: int,
    n_estimators: bool,
    criterion: int,
    max_depth: int or None,
    min_samples_split: int,
    bootstrap: bool,
    ccp_alpha: float,
    save_model: bool,
    only_fit: bool,
    cv_splits: int,
    shuffle: bool,
) -> None:

    mlflow.set_experiment("randfor")
    (features, target) = files.get_dataset(dataset_path)
    click.echo("Fitting model...")
    estimator = model.get_randfor(
        n_estimators=n_estimators,
        ccp_alpha=ccp_alpha,
        bootstrap=bootstrap,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
    )
    if not only_fit:
        with mlflow.start_run():
            metrics = crossvalidate.cross_validate_model(
                features=features,
                target=target,
                cv_splits=cv_splits,
                shuffle=shuffle,
                use_scaler=use_scaler,
                random_state=random_state,
                estimator=estimator,
            )
            mlflow_utils.log_mlflow_data(
                {
                    "n_estimators": n_estimators,
                    "ccp_alpha": ccp_alpha,
                    "bootstrap": bootstrap,
                    "max_depth": max_depth,
                    "criterion": criterion,
                    "min_samples_split": min_samples_split,
                },
                metrics,
            )

    if save_model:
        estimator.fit(features, target)
        files.save_the_model(estimator, use_scaler, save_model_path)
    click.echo("Finished.")
