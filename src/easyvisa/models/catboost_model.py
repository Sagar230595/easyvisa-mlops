from ..config import CATEGORICAL_FEATURES
from .base import BaseModel


class CatBoostModel(BaseModel):
    name = "catboost"
    flavor = "catboost"

    def suggest_params(self, trial):
        return dict(
            iterations=trial.suggest_int("iterations", 200, 1500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            border_count=trial.suggest_int("border_count", 32, 255),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
        )

    def build(self, params, seed):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            cat_features=CATEGORICAL_FEATURES, loss_function="Logloss",
            eval_metric="AUC", random_seed=seed, verbose=0, **params,
        )
