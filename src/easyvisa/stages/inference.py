"""Stage 04 - batch inference with the champion model (any flavor)."""
import os
from datetime import datetime

from ..base import PipelineStage
from ..config import FEATURES, CATEGORICAL_FEATURES
from ..models import ChampionModel


class BatchInferenceStage(PipelineStage):
    name = "inference"

    def __init__(self, cfg, input_table: str = None):
        super().__init__(cfg)
        self.input_table = input_table

    def run(self):
        src = self.input_table or self.cfg.fqn(self.cfg.features_table)
        pdf = self.spark.table(src).toPandas()
        for c in CATEGORICAL_FEATURES:
            pdf[c] = pdf[c].astype(str)

        champion = ChampionModel(self.cfg)
        self.log.info("Champion model=%s version=%s", champion.model_name, champion.version)

        proba = champion.predict_proba(pdf[FEATURES])
        pdf["visa_approval_proba"] = proba
        pdf["visa_prediction"] = (proba >= 0.5).astype(int)
        pdf["case_status_predicted"] = pdf["visa_prediction"].map({1: "Certified", 0: "Denied"})
        pdf["model_name"] = champion.model_name
        pdf["model_version"] = champion.version
        pdf["scored_at"] = datetime.utcnow().isoformat()

        (
            self.spark.createDataFrame(pdf)
            .write.mode("overwrite").option("overwriteSchema", "true")
            .saveAsTable(self.cfg.fqn(self.cfg.predictions_table))
        )
        self.log.info("Wrote predictions table %s", self.cfg.fqn(self.cfg.predictions_table))

        os.makedirs(self.cfg.exports_volume, exist_ok=True)
        fname = (f"easyvisa_predictions_v{champion.version}_"
                 f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
        path = os.path.join(self.cfg.exports_volume, fname)
        pdf.to_csv(path, index=False)
        self.log.info("Exported CSV %s", path)
        return path
