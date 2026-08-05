"""Stage 02 - preprocessing + feature engineering (Spark)."""
from ..base import PipelineStage
from ..config import (
    CATEGORICAL_FEATURES,
    LABEL_COL,
    NUMERIC_FEATURES,
    POSITIVE_CLASS,
    TARGET_COL,
    WAGE_MULTIPLIER,
)


class Preprocessor:
    """Data cleaning transforms."""

    def __init__(self, cfg):
        self.cfg = cfg

    def clean(self, df):
        from pyspark.sql import functions as F
        if "case_id" in df.columns:
            df = df.drop("case_id")
        df = df.withColumn("no_of_employees", F.abs(F.col("no_of_employees")))
        df = df.dropDuplicates()
        df = df.filter(F.col(TARGET_COL).isin(POSITIVE_CLASS, "Denied"))
        return df


class FeatureBuilder:
    """Turns a cleaned bronze frame into the model-ready feature table."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.pre = Preprocessor(cfg)

    def add_company_age(self, df):
        from pyspark.sql import functions as F
        age = F.greatest(F.lit(self.cfg.reference_year) - F.col("yr_of_estab"), F.lit(0))
        return df.withColumn("company_age", age).drop("yr_of_estab")

    @staticmethod
    def add_annual_wage(df):
        from pyspark.sql import functions as F
        mapping = []
        for unit, mult in WAGE_MULTIPLIER.items():
            mapping += [F.lit(unit), F.lit(float(mult))]
        factor = F.coalesce(F.create_map(*mapping)[F.col("unit_of_wage")], F.lit(1.0))
        return df.withColumn("annual_wage", F.col("prevailing_wage") * factor).drop("prevailing_wage")

    @staticmethod
    def encode_target(df):
        from pyspark.sql import functions as F
        return df.withColumn(
            LABEL_COL, F.when(F.col(TARGET_COL) == POSITIVE_CLASS, F.lit(1)).otherwise(F.lit(0))
        )

    def build(self, df):
        out = self.add_company_age(self.pre.clean(df))
        out = self.encode_target(self.add_annual_wage(out))
        keep = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, LABEL_COL]
        return out.select(*[c for c in keep if c in out.columns])


class FeatureEngineeringStage(PipelineStage):
    name = "feature_engineering"

    def run(self):
        bronze = self.spark.table(self.cfg.fqn(self.cfg.bronze_table))
        feats = FeatureBuilder(self.cfg).build(bronze)
        target = self.cfg.fqn(self.cfg.features_table)
        (feats.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target))
        self.log.info("Wrote features table %s (%s rows)", target, feats.count())
        return target
