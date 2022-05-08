import click.testing
import pytest

from sklearn.datasets import make_classification

from mpd import train,files


@pytest.fixture
def runner():
    return click.testing.CliRunner()

@pytest.fixture
def mock_get_dataset(mocker):
    mock = mocker.patch("files.get_dataset")
    mock.return_value = make_classification()
    return mock

def test_train(runner,mock_get_dataset):
   result = runner.invoke(train.train)
   assert result.exit_code == 0