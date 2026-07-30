"""Infrastructure services: logging, Spark session, MLflow / UC registry."""
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
    """Single access point for the active SparkSession."""

    _spark = None

    @classmethod
    def get(cls):
        if cls._spark is None:
            from pyspark.sql import SparkSession
            cls._spark = SparkSession.builder.getOrCreate()
        return cls._spark


class MLflowManager:
    """Wraps MLflow experiment setup and Unity Catalog registry access."""

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
        """Resolve the champion alias, else the latest version.

        Returns (model_uri, model_version).
        """
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
