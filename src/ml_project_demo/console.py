import click

from . import __version__
from data.report import makeReport

@click.command()
@click.version_option(version=__version__)
def main():
    """Menu"""
    #click.echo("Hello, world!")
    makeReport()