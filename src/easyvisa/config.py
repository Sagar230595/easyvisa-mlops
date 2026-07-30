"""Central configuration for the EasyVisa MLOps project.

Nothing here imports pyspark / catboost, so it is safe to import in CI.
"""
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional
import os

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# --- Feature contract -------------------------------------------------------
NUMERIC_FEATURES: List[str] = ["no_of_employees", "company_age", "annual_wage"]
CATEGORICAL_FEATURES: List[str] = [
    "continent",
    "education_of_employee",
    "has_job_experience",
    "requires_job_training",
    "region_of_employment",
    "unit_of_wage",
    "full_time_position",
]
FEATURES: List[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL: str = "case_status"
LABEL_COL: str = "case_status_label"
POSITIVE_CLASS: str = "Certified"

WAGE_MULTIPLIER: Dict[str, float] = {"Hour": 2080.0, "Week": 52.0, "Month": 12.0, "Year": 1.0}

# Model families tuned with Optuna and compared during training.
DEFAULT_CANDIDATE_MODELS: List[str] = [
    "catboost",
    "lightgbm",
    "xgboost",
    "random_forest",
    "logistic_regression",
]

_CONF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "conf"
)


@dataclass
class Config:
    env: str = "dev"
    catalog: str = "onedatalakehouse_datascience_we"
    schema: str = "easyvisa_dev"

    raw_file: str = "EasyVisa.csv"
    reference_year: int = 2016

    bronze_table: str = "easyvisa_bronze"
    features_table: str = "easyvisa_features"
    reference_table: str = "easyvisa_drift_reference"
    drift_report_table: str = "easyvisa_drift_report"
    predictions_table: str = "easyvisa_predictions"

    experiment_path: str = "/Shared/easyvisa_visa_approval_experiments"
    model_base_name: str = "easyvisa_visa_approval"
    champion_alias: str = "champion"

    # Training / tuning
    candidate_models: List[str] = field(default_factory=lambda: list(DEFAULT_CANDIDATE_MODELS))
    n_trials: int = 30                 # Optuna trials PER model family
    random_seed: int = 42
    test_size: float = 0.2

    @property
    def raw_volume(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw"

    @property
    def raw_path(self) -> str:
        return f"{self.raw_volume}/{self.raw_file}"

    @property
    def exports_volume(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/exports"

    @property
    def model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.model_base_name}"

    def fqn(self, table: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{table}`"

    @classmethod
    def load(cls, env: str = "dev", overrides: Optional[dict] = None) -> "Config":
        data: dict = {}
        path = os.path.join(_CONF_DIR, f"{env}.yml")
        if yaml is not None and os.path.exists(path):
            with open(path) as fh:
                data = yaml.safe_load(fh) or {}
        data["env"] = env
        if overrides:
            data.update({k: v for k, v in overrides.items() if v is not None})
        allowed = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in allowed})
