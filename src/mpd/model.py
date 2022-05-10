from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_model(model_type, random_state):
    if model_type is logreg_type:
        return get_logreg(random_state)
    if model_type is randfor_type:
        return get_randfor(random_state)


def get_logreg(random_state, max_iter=500, c=1, penalty="l2"):
    if penalty == ("l2" or "none"):
        solver = "lbfgs"
    else:
        solver = "saga"
    ratio = None
    if penalty == "elasticnet":
        ratio = 0.5
    model = LogisticRegression(
        random_state=random_state,
        max_iter=max_iter,
        C=c,
        penalty=penalty,
        solver=solver,
        l1_ratio=ratio,
    )
    return model


def get_randfor(
    random_state,
    n_estimators=100,
    criterion="gini",
    max_depth=None,
    min_samples_split=2,
    bootstrap=True,
    ccp_alpha=0,
):
    model = RandomForestClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha,
    )
    return model


def create_pipeline(estimator, use_scaler):
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(("estimator", estimator))
    return Pipeline(pipeline_steps)


logreg_type = type(LogisticRegression())
randfor_type = type(RandomForestClassifier())
model_types = {"logreg": logreg_type, "randfor": randfor_type}
