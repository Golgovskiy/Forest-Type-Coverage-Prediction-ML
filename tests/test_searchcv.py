import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from mpd import searchcv


# from sklearn.datasets import make_classification


# @pytest.fixture
# def runner():
#     return click.testing.CliRunner()


# @pytest.fixture
# def mock_get_dataset(mocker):
#     mock = mocker.patch("files.get_dataset")
#     mock.return_value = make_classification()
#     return mock

# def test_train(runner,mock_get_dataset):
#    result = runner.invoke(train.train)
#    assert result.exit_code == 0

def test_get_search_space_logreg_grid():
    result = searchcv.get_search(LogisticRegression(), "grid")
    assert result.param_grid is searchcv.logreg_space


def test_get_search_space_randfor_grid():
    result = searchcv.get_search(RandomForestClassifier(), "grid")
    assert result.param_grid is searchcv.randfor_space


def test_get_search_space_logreg_random():
    result = searchcv.get_search(LogisticRegression(), "random")
    assert result.param_distributions is searchcv.logreg_space


def test_get_search_space_randfor_random():
    result = searchcv.get_search(RandomForestClassifier(), "random")
    assert result.param_distributions is searchcv.randfor_space


def test_get_search_space_fail():
    with pytest.raises(KeyError):
        searchcv.get_search(GridSearchCV(None, {}), "random")


def test_get_search_grid():
    result = searchcv.get_search(RandomForestClassifier(), "grid")
    assert isinstance(result, GridSearchCV)


def test_get_search_random():
    result = searchcv.get_search(RandomForestClassifier(), "random")
    assert isinstance(result, RandomizedSearchCV)
