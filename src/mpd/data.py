# import logging
from pathlib import Path
from typing import Tuple

import click
import pandas as pd


def get_dataset(
    csv_path: Path, random_state: int, k_folds: int
) -> Tuple[pd.DataFrame, pd.Series]:

    dataset = pd.read_csv(csv_path)
    
    click.echo(f"Dataset shape: {dataset.shape}.")
    
    features = dataset.drop("target", axis=1)
    target = dataset["target"]
    
    return features, target