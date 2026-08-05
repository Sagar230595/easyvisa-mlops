import pandas as pd

from easyvisa.models import ModelRegistry


class FakeTrial:
    def suggest_int(self, name, low, high, **k):
        return low

    def suggest_float(self, name, low, high, **k):
        return low

    def suggest_categorical(self, name, choices):
        return choices[0]


def _sample_row():
    return pd.DataFrame({
        "no_of_employees": [10], "company_age": [5], "annual_wage": [50000.0],
        "continent": ["Asia"], "education_of_employee": ["Master's"],
        "has_job_experience": ["Y"], "requires_job_training": ["N"],
        "region_of_employment": ["West"], "unit_of_wage": ["Year"],
        "full_time_position": ["Y"],
    })


def test_registry_has_five_models():
    reg = ModelRegistry()
    assert set(reg.names) == {
        "catboost", "lightgbm", "xgboost", "random_forest", "logistic_regression"
    }


def test_each_model_suggests_params_and_has_valid_flavor():
    for m in ModelRegistry().selected():
        params = m.suggest_params(FakeTrial())
        assert isinstance(params, dict) and params
        assert m.flavor in {"catboost", "lightgbm", "xgboost", "sklearn"}


def test_booster_prep_uses_category_dtype():
    m = ModelRegistry().get("lightgbm")
    assert str(m.prep_features(_sample_row())["continent"].dtype) == "category"


def test_catboost_prep_not_category():
    m = ModelRegistry().get("catboost")
    assert str(m.prep_features(_sample_row())["continent"].dtype) != "category"


def test_from_config_subset():
    class Cfg:
        candidate_models = ["catboost", "xgboost"]
    reg = ModelRegistry.from_config(Cfg())
    assert [m.name for m in reg.selected()] == ["catboost", "xgboost"]
