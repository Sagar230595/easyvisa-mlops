from .random_forest_model import _preprocessor
from .base import BaseModel


class LogisticRegressionModel(BaseModel):
    name = "logistic_regression"
    flavor = "sklearn"

    def suggest_params(self, trial):
        return dict(C=trial.suggest_float("C", 1e-3, 10.0, log=True))

    def build(self, params, seed):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("prep", _preprocessor(True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed, **params)),
        ])
