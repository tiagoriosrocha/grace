"""Modular, auditable local fact-checking pipeline."""

from .conditions import Condition, extract_conditions
from .aggregation import aggregate_conditions

__all__ = ["Condition", "extract_conditions", "aggregate_conditions"]
