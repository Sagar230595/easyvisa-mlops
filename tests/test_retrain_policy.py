from easyvisa.stages.monitoring import RetrainPolicy


class Cfg:
    min_auc_threshold = 0.75


def _policy():
    return RetrainPolicy(Cfg())


def test_no_retrain_when_stable():
    retrain, reasons = _policy().decide(data_drift=False, model_drift=False, current_auc=0.9)
    assert retrain is False and reasons == []


def test_retrain_on_data_drift():
    retrain, reasons = _policy().decide(True, False, 0.9)
    assert retrain and "data_drift" in reasons


def test_retrain_on_model_drift():
    retrain, reasons = _policy().decide(False, True, 0.9)
    assert retrain and "model_drift" in reasons


def test_retrain_when_auc_below_floor():
    retrain, reasons = _policy().decide(False, False, 0.70)
    assert retrain and any("auc_below" in r for r in reasons)


def test_no_retrain_when_auc_unknown():
    retrain, reasons = _policy().decide(False, False, None)
    assert retrain is False
