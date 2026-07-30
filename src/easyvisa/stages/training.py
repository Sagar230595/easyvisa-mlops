"""Stage 03 - multi-model training orchestrated by the Trainer class."""
from ..base import PipelineStage
from ..config import FEATURES, CATEGORICAL_FEATURES, LABEL_COL
from ..infra import MLflowManager
from ..models import ModelRegistry


class Trainer:
    """Runs Optuna tuning across every candidate model, logs to MLflow, and
    registers/promotes the best one in Unity Catalog.
    """

    def __init__(self, cfg, spark, log):
        self.cfg = cfg
        self.spark = spark
        self.log = log
        self.registry = ModelRegistry.from_config(cfg)
        self.mlflow_mgr = MLflowManager(cfg)

    def load_pandas(self):
        pdf = self.spark.table(self.cfg.fqn(self.cfg.features_table)).toPandas()
        for c in CATEGORICAL_FEATURES:
            pdf[c] = pdf[c].astype(str)
        return pdf

    def split(self, pdf):
        from sklearn.model_selection import train_test_split
        X, y = pdf[FEATURES], pdf[LABEL_COL]
        return train_test_split(
            X, y, test_size=self.cfg.test_size, stratify=y, random_state=self.cfg.random_seed
        )

    @staticmethod
    def holdout_metrics(y_true, proba):
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
        )
        pred = (proba >= 0.5).astype(int)
        return dict(
            test_auc=float(roc_auc_score(y_true, proba)),
            test_accuracy=float(accuracy_score(y_true, pred)),
            test_f1=float(f1_score(y_true, pred)),
            test_precision=float(precision_score(y_true, pred)),
            test_recall=float(recall_score(y_true, pred)),
        )

    def _tune(self, model, X_tr, y_tr):
        import optuna
        import mlflow
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.cfg.random_seed)
        )

        def objective(trial):
            params = model.suggest_params(trial)
            score = model.cv_auc(params, X_tr, y_tr, self.cfg.random_seed)
            mlflow.log_metric("trial_cv_auc", score, step=trial.number)   # every trial logged
            return score

        study.optimize(objective, n_trials=self.cfg.n_trials)
        return study

    def run(self):
        import optuna
        import mlflow
        from mlflow.models.signature import infer_signature

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.mlflow_mgr.setup()

        pdf = self.load_pandas()
        X_tr, X_te, y_tr, y_te = self.split(pdf)
        X = pdf[FEATURES]

        leaderboard = []  # (name, test_auc, run_id, est, model)
        with mlflow.start_run(run_name="multimodel_training"):
            mlflow.log_param("candidate_models", ",".join(self.cfg.candidate_models))
            mlflow.log_param("n_trials_per_model", self.cfg.n_trials)
            mlflow.log_param("features", ",".join(FEATURES))

            for model in self.registry.selected():
                self.log.info("Tuning %s ...", model.name)
                with mlflow.start_run(run_name=model.name, nested=True) as child:
                    study = self._tune(model, X_tr, y_tr)
                    best = study.best_params
                    est = model.fit(best, X_tr, y_tr, self.cfg.random_seed)
                    proba = model.predict_proba(est, X_te)
                    metrics = self.holdout_metrics(y_te, proba)
                    metrics["cv_auc"] = float(study.best_value)

                    mlflow.set_tag("model_name", model.name)
                    mlflow.set_tag("flavor", model.flavor)
                    mlflow.log_param("model_name", model.name)
                    mlflow.log_param("flavor", model.flavor)
                    mlflow.log_params({f"param_{k}": v for k, v in best.items()})
                    mlflow.log_metrics(metrics)

                    study.trials_dataframe().to_csv("/tmp/trials.csv", index=False)
                    mlflow.log_artifact("/tmp/trials.csv")

                    sig = infer_signature(model.prep_features(X_tr), proba)
                    model.log_model(est, "model", sig)

                    leaderboard.append((model.name, metrics["test_auc"], child.info.run_id, est, model))
                    self.log.info("%s test AUC=%.4f (cv=%.4f)",
                                  model.name, metrics["test_auc"], study.best_value)

            leaderboard.sort(key=lambda t: t[1], reverse=True)
            best_name, best_auc, best_run_id, best_est, best_model = leaderboard[0]
            for nm, auc, _, _, _ in leaderboard:
                mlflow.log_metric(f"holdout_auc_{nm}", auc)
            mlflow.log_param("winning_model", best_name)
            mlflow.log_metric("best_holdout_auc", best_auc)

        # Register + promote the winner
        client = self.mlflow_mgr.client
        mv = mlflow.register_model(f"runs:/{best_run_id}/model", self.cfg.model_name)
        client.set_registered_model_alias(self.cfg.model_name, self.cfg.champion_alias, mv.version)
        client.set_model_version_tag(self.cfg.model_name, mv.version, "model_name", best_name)
        client.set_model_version_tag(self.cfg.model_name, mv.version, "flavor", best_model.flavor)
        self.log.info("Champion %s (auc=%.4f) -> %s v%s @%s",
                      best_name, best_auc, self.cfg.model_name, mv.version, self.cfg.champion_alias)

        # Drift reference matrix from the champion
        ref = pdf.copy()
        ref["prediction_proba"] = best_model.predict_proba(best_est, X)
        ref["run_id"] = best_run_id
        ref["model_name"] = best_name
        (
            self.spark.createDataFrame(ref)
            .write.mode("overwrite").option("overwriteSchema", "true")
            .saveAsTable(self.cfg.fqn(self.cfg.reference_table))
        )
        return {"winning_model": best_name, "best_holdout_auc": best_auc,
                "leaderboard": [(n, a) for n, a, _, _, _ in leaderboard]}


class ModelTrainingStage(PipelineStage):
    name = "model_training"

    def run(self):
        return Trainer(self.cfg, self.spark, self.log).run()
