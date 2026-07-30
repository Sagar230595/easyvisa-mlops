"""Model strategy registry + champion loader."""
from typing import Dict, List, Optional

from .base import BaseModel
from .catboost_model import CatBoostModel
from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .random_forest_model import RandomForestModel
from .logistic_regression_model import LogisticRegressionModel

_ALL_MODEL_CLASSES = [
    CatBoostModel,
    LightGBMModel,
    XGBoostModel,
    RandomForestModel,
    LogisticRegressionModel,
]


class ModelRegistry:
    """Holds the available model strategies and the selected subset."""

    def __init__(self, names: Optional[List[str]] = None):
        self._all: Dict[str, BaseModel] = {cls().name: cls() for cls in _ALL_MODEL_CLASSES}
        self.names: List[str] = names or list(self._all.keys())

    @classmethod
    def from_config(cls, cfg) -> "ModelRegistry":
        return cls(list(cfg.candidate_models))

    def get(self, name: str) -> BaseModel:
        return self._all[name]

    def selected(self) -> List[BaseModel]:
        return [self._all[n] for n in self.names]


class ChampionModel:
    """Loads the production (champion) model regardless of its flavor and
    exposes a uniform ``predict_proba``.
    """

    def __init__(self, cfg):
        from ..infra import MLflowManager
        uri, mv = MLflowManager(cfg).production_model_version()
        tags = getattr(mv, "tags", None) or {}
        self.model_name = tags.get("model_name", "catboost")
        self.strategy = ModelRegistry().get(self.model_name)
        self.estimator = self.strategy.load_model(uri)
        self.version = mv.version
        self.run_id = mv.run_id

    def predict_proba(self, X):
        return self.strategy.predict_proba(self.estimator, X)


__all__ = [
    "BaseModel", "ModelRegistry", "ChampionModel",
    "CatBoostModel", "LightGBMModel", "XGBoostModel",
    "RandomForestModel", "LogisticRegressionModel",
]
