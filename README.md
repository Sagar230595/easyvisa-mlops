# EasyVisa MLOps

End-to-end, modular MLOps project on Databricks that predicts US visa
certification (`case_status`) from the EasyVisa dataset using CatBoost, with
Optuna tuning, MLflow tracking, Unity Catalog model registry, and drift
monitoring. CI/CD is driven from GitHub via Databricks Asset Bundles.

## Pipeline stages
1. `data_ingestion`      raw CSV (Volume) -> bronze Delta
2. `feature_engineering` cleaning + feature build -> features Delta
3. `train`               Optuna + CV CatBoost -> MLflow + UC registry (@champion)
4. `inference`           champion model -> predictions Delta + CSV export
5. `monitoring`          data & model drift report (informational)

## Layout (OOP)
```
src/easyvisa/
  config.py            Config dataclass + feature contract & constants
  transforms.py        pure row-level helpers (unit-tested)
  infra.py             SparkProvider, MLflowManager, logging
  base.py              PipelineStage (abstract base)
  models/              strategy pattern: BaseModel + one class per family,
                       ModelRegistry, ChampionModel
  stages/              PipelineStage subclasses:
                       DataIngestionStage, FeatureEngineeringStage (+FeatureBuilder,
                       Preprocessor), ModelTrainingStage (+Trainer),
                       BatchInferenceStage, DriftMonitoringStage (+DriftDetector)
workflows/             job entrypoints that instantiate the stage objects
conf/ resources/ tests/ .github/ databricks.yml   (as before)
```

See `IMPLEMENTATION_STEPS.txt` for the full setup and deployment walkthrough.

## Local dev
```
pip install -r requirements-dev.txt
pytest -q
ruff check src tests
```
