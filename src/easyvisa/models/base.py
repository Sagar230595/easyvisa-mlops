"""BaseModel strategy: each model family encapsulates its own search space,
estimator construction, categorical handling, CV, and MLflow flavor.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from ..config import NUMERIC_FEATURES, CATEGORICAL_FEATURES


class BaseModel(ABC):
    name: str = "base"
    flavor: str = "sklearn"          # MLflow flavor used to log/load

    # --- required per family ---
    @abstractmethod
    def suggest_params(self, trial) -> dict:
        ...

    @abstractmethod
    def build(self, params: dict, seed: int):
        ...

    # --- shared behaviour (overridable) ---
    def prep_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Default: categoricals as strings (CatBoost + sklearn pipelines)."""
        X = X.copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype(str)
        return X

    def cv_auc(self, params: dict, X, y, seed: int, n_splits: int = 5) -> float:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import roc_auc_score

        Xp = self.prep_features(X)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = []
        for tr, va in skf.split(Xp, y):
            est = self.build(params, seed)
            est.fit(Xp.iloc[tr], y.iloc[tr])
            proba = est.predict_proba(Xp.iloc[va])[:, 1]
            scores.append(roc_auc_score(y.iloc[va], proba))
        return float(np.mean(scores))

    def fit(self, params: dict, X, y, seed: int):
        est = self.build(params, seed)
        est.fit(self.prep_features(X), y)
        return est

    def predict_proba(self, est, X):
        return est.predict_proba(self.prep_features(X))[:, 1]

    def log_model(self, est, artifact_path: str, signature):
        import mlflow
        getattr(mlflow, self.flavor).log_model(est, artifact_path=artifact_path, signature=signature)

    def load_model(self, uri: str):
        import mlflow
        return getattr(mlflow, self.flavor).load_model(uri)


class _CategoryDtypeMixin:
    """For libraries (LightGBM/XGBoost) that consume pandas 'category' dtype."""

    def prep_features(self, X):
        X = X.copy()
        for c in CATEGORICAL_FEATURES:
            X[c] = X[c].astype("category")
        return X
