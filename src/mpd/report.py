import pandas as pd
from pathlib import Path
import click

from pandas_profiling import ProfileReport


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True),
    show_default=True,
)
@click.option(
    "-s",
    "--save-to",
    default="data/report.html",
    type=click.Path(exists=False),
    show_default=True,
)
def makeReport(dataset_path: Path, save_to: Path) -> None:
    df = pd.read_csv(dataset_path)
    profile = ProfileReport(
        df, title="Pandas Profiling Report", missing_diagrams={"Count": False}
    )
    profile.to_file(save_to)
    click.echo("Report was saved to: " + str(save_to))