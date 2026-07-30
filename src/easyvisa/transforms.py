"""Pure, dependency-light transform helpers.

These contain the row-level business logic used by the Spark feature pipeline.
Keeping them free of pyspark/catboost makes them fast and trivial to unit test.
"""
from .config import WAGE_MULTIPLIER, POSITIVE_CLASS


def wage_multiplier(unit_of_wage: str) -> float:
    """Annualisation factor for a wage unit; unknown units default to yearly."""
    return WAGE_MULTIPLIER.get(unit_of_wage, 1.0)


def annualise_wage(prevailing_wage: float, unit_of_wage: str) -> float:
    return float(prevailing_wage) * wage_multiplier(unit_of_wage)


def company_age(year_of_estab: int, reference_year: int) -> int:
    """Age of company; clamped at 0 for future establishment years."""
    return max(int(reference_year) - int(year_of_estab), 0)


def fix_employee_count(no_of_employees: int) -> int:
    """A handful of rows carry negative counts (data entry errors)."""
    return abs(int(no_of_employees))


def encode_label(case_status: str) -> int:
    """Certified -> 1, anything else -> 0."""
    return 1 if case_status == POSITIVE_CLASS else 0
