"""Compatibility wrapper for split-schema evaluation workflows."""

from .eval_split import (
    compare_evaluation_methods,
    run_split_evaluation_categorical,
    run_split_evaluation_numeric,
)

__all__ = [
    "compare_evaluation_methods",
    "run_split_evaluation_categorical",
    "run_split_evaluation_numeric",
]
