"""Databricks job entrypoint (spark_python_task)."""
import argparse

from easyvisa.config import Config
from easyvisa.stages.monitoring import DriftMonitoringStage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev")
    parser.add_argument("--catalog", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--n_trials", type=int, default=None)
    parser.add_argument("--current_table", default=None)
    args = parser.parse_args()

    cfg = Config.load(
        args.env,
        overrides={"catalog": args.catalog, "schema": args.schema, "n_trials": args.n_trials},
    )
    DriftMonitoringStage(cfg, current_table=args.current_table).run()


if __name__ == "__main__":
    main()
