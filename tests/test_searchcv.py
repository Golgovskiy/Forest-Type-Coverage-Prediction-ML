import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from mpd import searchcv, model


# from sklearn.datasets import make_classification
randfor_type = model.randfor_type
logreg_type = model.logreg_type

def test_get_search_space_logreg_grid():
    result = searchcv.get_search(logreg_type, "grid")
    assert result.param_grid is searchcv.logreg_space


def test_get_search_space_randfor_grid():
    result = searchcv.get_search(randfor_type, "grid")
    assert result.param_grid is searchcv.randfor_space


def test_get_search_space_logreg_random():
    result = searchcv.get_search(logreg_type, "random")
    assert result.param_distributions is searchcv.logreg_space


def test_get_search_space_randfor_random():
    result = searchcv.get_search(randfor_type, "random")
    assert result.param_distributions is searchcv.randfor_space


def test_get_search_space_fail():
    with pytest.raises(KeyError):
        searchcv.get_search(GridSearchCV(None, {}), "random")


def test_get_search_grid():
    result = searchcv.get_search(randfor_type, "grid")
    assert isinstance(result, GridSearchCV)


def test_get_search_random():
    result = searchcv.get_search(randfor_type, "random")
    assert isinstance(result, RandomizedSearchCV)
