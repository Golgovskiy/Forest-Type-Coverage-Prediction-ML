from typing import Tuple
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame, Series

from mpd import model


def search(
    searcher: BaseEstimator, features: DataFrame, targets: Series, use_scaler: bool
) -> Tuple:  # type: ignore
    searcher.fit(X=features, y=targets.values.ravel())
    return (
        searcher[int(use_scaler)].best_estimator_,
        searcher[int(use_scaler)].best_params_,
    )


def get_search(
    model_type: type,
    search_type: str,
    random_state: int = 42,
    cv_splits: int = 5,
    shuffle: bool = True,
) -> BaseEstimator:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=shuffle, random_state=random_state)

    if search_type == "grid":
        search_cv = GridSearchCV(
            model.get_model(model_type=model_type, random_state=random_state),
            spaces[model_type],
            scoring="roc_auc_ovr",
            n_jobs=-1,
            cv=cv,
            refit=True,
        )
    if search_type == "random":
        search_cv = RandomizedSearchCV(
            model.get_model(model_type=model_type, random_state=random_state),
            spaces[model_type],
            scoring="roc_auc_ovr",
            n_jobs=-1,
            cv=cv,
            refit=True,
        )
    return search_cv


randfor_space = {
    "n_estimators": [5, 20, 50, 100, 500],
    "criterion": ["gini", "entropy"],
    "max_depth": [2, 5, 10, 20, 50, 100, 500, None],
    "min_samples_split": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    "bootstrap": [True, False],
    "ccp_alpha": [0, 1, 2, 5, 10, 30, 75, 100, 200],
}
logreg_space = {
    "C": [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
    "max_iter": [2000, 5000, 10000],
}
spaces = {model.randfor_type: randfor_space, model.logreg_type: logreg_space}
