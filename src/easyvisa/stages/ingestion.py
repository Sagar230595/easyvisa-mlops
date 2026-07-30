"""Stage 01 - ingest the raw EasyVisa CSV into a bronze Delta table."""
from ..base import PipelineStage


class DataIngestionStage(PipelineStage):
    name = "data_ingestion"

    COLUMN_MAP = {
        "Main_Fact_Date": "Posting_Date",   # harmless if absent
        "NS_Invoiced": "Invoiced_NS",
        "NS_Ordered": "Ordered_NS",
    }
    REQUIRED = ["case_status"]

    def read_raw(self):
        return (
            self.spark.read.option("header", True).option("inferSchema", True)
            .csv(self.cfg.raw_path)
        )

    def run(self):
        df = self.read_raw()
        n = df.count()
        self.log.info("Raw rows: %s | columns: %s", n, df.columns)

        self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{self.cfg.catalog}`.`{self.cfg.schema}`")
        target = self.cfg.fqn(self.cfg.bronze_table)
        (df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target))
        self.log.info("Wrote bronze table %s (%s rows)", target, n)
        return target
