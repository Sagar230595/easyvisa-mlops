from .base import BaseModel, _CategoryDtypeMixin


class XGBoostModel(_CategoryDtypeMixin, BaseModel):
    name = "xgboost"
    flavor = "xgboost"

    def suggest_params(self, trial):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 1200),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        )

    def build(self, params, seed):
        from xgboost import XGBClassifier
        return XGBClassifier(
            enable_categorical=True, tree_method="hist", eval_metric="auc",
            random_state=seed, verbosity=0, **params,
        )
