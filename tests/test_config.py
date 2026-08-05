from easyvisa.config import CATEGORICAL_FEATURES, FEATURES, Config


def test_defaults():
    cfg = Config()
    assert cfg.env == "dev"
    assert cfg.model_name.endswith("easyvisa_visa_approval")
    assert cfg.raw_path.endswith("EasyVisa.csv")


def test_fqn_backticks():
    cfg = Config(catalog="c", schema="07_s")
    assert cfg.fqn("t") == "`c`.`07_s`.`t`"


def test_feature_contract_no_overlap_with_label():
    assert "case_status_label" not in FEATURES
    assert all(c in FEATURES for c in CATEGORICAL_FEATURES)
