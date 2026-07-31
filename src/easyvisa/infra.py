"""Infrastructure services: logging, Spark, MLflow / UC registry, retraining."""
import logging


def get_logger(name: str = "easyvisa") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class SparkProvider:
    _spark = None

    @classmethod
    def get(cls):
        if cls._spark is None:
            from pyspark.sql import SparkSession
            cls._spark = SparkSession.builder.getOrCreate()
        return cls._spark


def get_dbutils(spark=None):
    """Return a DBUtils handle inside a job, or None when unavailable."""
    try:
        from pyspark.dbutils import DBUtils
        return DBUtils(spark or SparkProvider.get())
    except Exception:
        return None


class MLflowManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self) -> "MLflowManager":
        import mlflow
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(self.cfg.experiment_path)
        return self

    @property
    def client(self):
        from mlflow.tracking import MlflowClient
        return MlflowClient(registry_uri="databricks-uc")

    def production_model_version(self):
        name = self.cfg.model_name
        client = self.client
        try:
            mv = client.get_model_version_by_alias(name, self.cfg.champion_alias)
            return f"models:/{name}@{self.cfg.champion_alias}", mv
        except Exception:
            versions = client.search_model_versions(f"name='{name}'")
            if not versions:
                raise RuntimeError(f"No registered versions found for {name}")
            mv = max(versions, key=lambda v: int(v.version))
            return f"models:/{name}/{mv.version}", mv


class Retrainer:
    """Triggers the training pipeline job via the Databricks Jobs API."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _find_job_id(self, w):
        for job in w.jobs.list():
            name = (job.settings.name if job.settings else "") or ""
            if self.cfg.training_job_match in name:
                return job.job_id
        return None

    def trigger(self):
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        job_id = self._find_job_id(w)
        if job_id is None:
            raise RuntimeError(
                f"No job whose name contains '{self.cfg.training_job_match}' was found"
            )
        run = w.jobs.run_now(job_id=job_id)
        return run.run_id, job_id
