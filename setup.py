from setuptools import setup, find_packages

setup(
    name="easyvisa",
    version="0.4.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "catboost>=1.2",
        "lightgbm>=4.0",
        "xgboost>=2.0",
        "optuna>=3.6",
        "mlflow>=2.14",
        "scikit-learn>=1.3",
        "scipy>=1.11",
        "pandas>=2.0",
        "numpy>=1.24",
        "pyyaml>=6.0",
        "databricks-sdk>=0.30",
    ],
)
