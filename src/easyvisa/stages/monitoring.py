"""Stage 05 - data & model drift monitoring (informational only)."""
import numpy as np
import pandas as pd

from ..base import PipelineStage
from ..config import FEATURES, CATEGORICAL_FEATURES, LABEL_COL
from ..infra import MLflowManager
from ..models import ChampionModel


class DriftDetector:
    """PSI / KS based drift statistics."""

    def __init__(self, psi_threshold: float = 0.2):
        self.psi_threshold = psi_threshold

    @staticmethod
    def psi_numeric(expected, actual, bins: int = 10) -> float:
        expected = np.asarray(expected, float)
        actual = np.asarray(actual, float)
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        if len(expected) == 0 or len(actual) == 0:
            return float("nan")
        cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
        if len(cuts) < 3:
            return 0.0
        cuts[0], cuts[-1] = -np.inf, np.inf
        e = np.histogram(expected, cuts)[0] / len(expected)
        a = np.histogram(actual, cuts)[0] / len(actual)
        e = np.clip(e, 1e-6, None)
        a = np.clip(a, 1e-6, None)
        return float(np.sum((a - e) * np.log(a / e)))

    @staticmethod
    def psi_categorical(expected, actual) -> float:
        e = pd.Series(expected).astype(str).value_counts(normalize=True)
        a = pd.Series(actual).astype(str).value_counts(normalize=True)
        cats = e.index.union(a.index)
        e = e.reindex(cats).fillna(0).clip(lower=1e-6)
        a = a.reindex(cats).fillna(0).clip(lower=1e-6)
        return float(np.sum((a - e) * np.log(a / e)))

    def feature_drift(self, ref, cur) -> pd.DataFrame:
        from scipy.stats import ks_2samp
        rows = []
        for col in FEATURES:
            if col in CATEGORICAL_FEATURES:
                psi = self.psi_categorical(ref[col], cur[col])
                ksp = float("nan")
            else:
                psi = self.psi_numeric(ref[col], cur[col])
                ksp = float(ks_2samp(ref[col].dropna(), cur[col].dropna()).pvalue)
            drifted = bool(psi is not None and not np.isnan(psi) and psi > self.psi_threshold)
            rows.append(dict(feature=col, psi=psi, ks_pvalue=ksp, drifted=drifted))
        return pd.DataFrame(rows)


class DriftMonitoringStage(PipelineStage):
    name = "monitoring"

    def __init__(self, cfg, current_table: str = None,
                 psi_threshold: float = 0.2, auc_drop_threshold: float = 0.05):
        super().__init__(cfg)
        self.current_table = current_table
        self.detector = DriftDetector(psi_threshold)
        self.auc_drop_threshold = auc_drop_threshold

    def run(self):
        import mlflow
        from sklearn.metrics import roc_auc_score

        ref = self.spark.table(self.cfg.fqn(self.cfg.reference_table)).toPandas()
        cur = self.spark.table(self.current_table or self.cfg.fqn(self.cfg.features_table)).toPandas()
        for d in (ref, cur):
            for c in CATEGORICAL_FEATURES:
                d[c] = d[c].astype(str)

        drift_df = self.detector.feature_drift(ref, cur)
        data_drift = bool(drift_df["drifted"].any())

        MLflowManager(self.cfg).setup()
        champion = ChampionModel(self.cfg)
        cur_proba = champion.predict_proba(cur[FEATURES])
        pred_psi = self.detector.psi_numeric(ref["prediction_proba"], cur_proba)
        prediction_drift = bool(not np.isnan(pred_psi) and pred_psi > self.detector.psi_threshold)

        baseline_auc = None
        if champion.run_id:
            baseline_auc = mlflow.get_run(champion.run_id).data.metrics.get("test_auc")
        current_auc = None
        if LABEL_COL in cur.columns and cur[LABEL_COL].nunique() > 1:
            current_auc = float(roc_auc_score(cur[LABEL_COL], cur_proba))
        performance_drift = bool(
            baseline_auc is not None and current_auc is not None
            and (baseline_auc - current_auc) > self.auc_drop_threshold
        )
        model_drift = bool(prediction_drift or performance_drift)

        with mlflow.start_run(run_name="drift_monitoring"):
            mlflow.log_param("monitored_model_name", champion.model_name)
            mlflow.log_param("monitored_model_version", champion.version)
            for _, r in drift_df.iterrows():
                if pd.notna(r["psi"]):
                    mlflow.log_metric(f"psi_{r['feature']}", float(r["psi"]))
            mlflow.log_metric("prediction_psi", 0.0 if np.isnan(pred_psi) else float(pred_psi))
            if baseline_auc is not None:
                mlflow.log_metric("baseline_auc", float(baseline_auc))
            if current_auc is not None:
                mlflow.log_metric("current_auc", float(current_auc))
            mlflow.log_metric("data_drift", int(data_drift))
            mlflow.log_metric("model_drift", int(model_drift))

        report = drift_df.copy()
        report["monitored_model_name"] = champion.model_name
        report["prediction_psi"] = pred_psi
        report["baseline_auc"] = baseline_auc
        report["current_auc"] = current_auc
        report["data_drift"] = data_drift
        report["model_drift"] = model_drift
        report["evaluated_at"] = pd.Timestamp.now()
        (
            self.spark.createDataFrame(report)
            .write.mode("append").option("mergeSchema", "true")
            .saveAsTable(self.cfg.fqn(self.cfg.drift_report_table))
        )

        print("=" * 56)
        print("EASYVISA DRIFT MONITORING SUMMARY")
        print("=" * 56)
        print(f"Monitored model : {champion.model_name} v{champion.version}")
        print(f"Data drift  : {'DETECTED' if data_drift else 'none'}")
        if data_drift:
            print("  Drifting features:", drift_df.loc[drift_df['drifted'], 'feature'].tolist())
        print(f"Model drift : {'DETECTED' if model_drift else 'none'}")
        if not (data_drift or model_drift):
            print("No significant drift. Model and data look stable.")
        else:
            print("Drift detected - consider a retraining run.")
        print("=" * 56)
        return dict(data_drift=data_drift, model_drift=model_drift)
