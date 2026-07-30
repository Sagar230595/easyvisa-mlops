from ..config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from .base import BaseModel


def _preprocessor(scale_numeric: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num, NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


class RandomForestModel(BaseModel):
    name = "random_forest"
    flavor = "sklearn"

    def suggest_params(self, trial):
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 200, 800),
            max_depth=trial.suggest_int("max_depth", 4, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        )

    def build(self, params, seed):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("prep", _preprocessor(False)),
            ("clf", RandomForestClassifier(random_state=seed, n_jobs=-1, **params)),
        ])
