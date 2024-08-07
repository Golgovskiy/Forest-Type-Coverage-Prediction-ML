# Forest Cover Type Prediction
Graduation ML project for RS School

This project uses [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset.

MLFLow experiment example:

![Alt text](etc/mlflow.png?raw=true)

## Usage
This package allows you to train model for predicting the forest coverage based on geographical features.
1. Clone this repository to your machine.
2. Download [Forest Cover Type Prediction](https://www.kaggle.com/competitions/forest-cover-type-prediction/data) dataset, save train.csv locally (default path is *data/train.csv* in repository's root).
3. Make sure Python 3.8 and [Poetry](https://python-poetry.org/docs/) are installed on your machine.
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run any of the train* files with the following command:
```sh
poetry run train -d <path to csv with data> -st <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
6. Run MLF~~~~low UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
You can run report file to generate a pandas profiling report.html (and save to place of you choice with options of CLI)
```
poetry run report -d <path to csv with data> -s <path to save report>
```

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```
Now you can use developer instruments, e.g. pytest:
```
poetry run pytest
```
More conveniently, to run all sessions of testing and formatting in a single command, install and use [nox](https://nox.thea.codes/en/stable/): 
```
nox [-r]
```
Format your code with [black](https://github.com/psf/black) by using either nox or poetry:
```
nox -[r]s black
poetry run black src tests noxfile.py
```
The end result must be:

![Alt text](etc/result.png?raw=true)
