from .base import BaseModel, _CategoryDtypeMixin


class LightGBMModel(_CategoryDtypeMixin, BaseModel):
    name = "lightgbm"
    flavor = "lightgbm"

    def suggest_params(self, trial):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            num_leaves=trial.suggest_int("num_leaves", 15, 255),
            max_depth=trial.suggest_int("max_depth", -1, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        )

    def build(self, params, seed):
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, verbose=-1, **params)
