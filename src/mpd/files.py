from typing import Tuple
from pathlib import Path
from joblib import dump

from sklearn.impute import KNNImputer
from sklearn.base import BaseEstimator
import click
import pandas as pd

from mpd import model


def get_dataset(
    csv_path: Path, cleanup: bool = False
) -> Tuple[pd.DataFrame, pd.Series]:
    dataset = pd.read_csv(csv_path)
    if cleanup:
        dataset = clean(dataset)

    click.echo(f"Dataset shape: {dataset.shape}.")

    features = dataset.iloc[:, :-1]
    target = dataset.iloc[:, -1:]

    return features, target


def save_the_model(estimator: BaseEstimator, use_scaler: bool, path: Path) -> None:
    dump(model.create_pipeline(estimator=estimator, use_scaler=use_scaler), path)
    click.echo(f"Model is saved to '{path}'.")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if len(df[df.isna()]) / len(df) < 0.25:
        df = df.dropna()
    else:
        imp = KNNImputer(n_neighbors=5, weights="distance")
        df = imp.fit_transform(df)
    return df
