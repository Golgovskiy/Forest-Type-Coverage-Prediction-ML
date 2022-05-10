from typing import Tuple
from pathlib import Path
from joblib import dump

import click
import pandas as pd

from mpd import model


def get_dataset(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:

    dataset = pd.read_csv(csv_path)

    click.echo(f"Dataset shape: {dataset.shape}.")

    features = dataset.iloc[:, :-1]
    target = dataset.iloc[:, -1:]

    return features, target


def save_the_model(estimator, use_scaler, path):
    dump(model.create_pipeline(estimator=estimator, use_scaler=use_scaler), path)
    click.echo(f"Model is saved to '{path}'.")
