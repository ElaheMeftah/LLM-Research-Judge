"""Compatibility wrapper for the full-schema evaluation workflow."""

from .eval_complete import create_evaluation_prompt, run_complete_evaluation

__all__ = ["create_evaluation_prompt", "run_complete_evaluation"]
