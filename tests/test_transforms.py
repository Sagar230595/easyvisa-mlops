from easyvisa.transforms import (
    annualise_wage,
    company_age,
    encode_label,
    fix_employee_count,
    wage_multiplier,
)


def test_wage_multiplier_known_units():
    assert wage_multiplier("Year") == 1.0
    assert wage_multiplier("Hour") == 2080.0
    assert wage_multiplier("Week") == 52.0
    assert wage_multiplier("Month") == 12.0


def test_wage_multiplier_unknown_defaults_yearly():
    assert wage_multiplier("Fortnight") == 1.0


def test_annualise_wage():
    assert annualise_wage(10.0, "Hour") == 20800.0
    assert annualise_wage(50000.0, "Year") == 50000.0


def test_company_age():
    assert company_age(2000, 2016) == 16
    assert company_age(2016, 2016) == 0


def test_company_age_future_clamped():
    assert company_age(2020, 2016) == 0


def test_fix_employee_count_handles_negatives():
    assert fix_employee_count(-26) == 26
    assert fix_employee_count(14513) == 14513


def test_encode_label():
    assert encode_label("Certified") == 1
    assert encode_label("Denied") == 0
