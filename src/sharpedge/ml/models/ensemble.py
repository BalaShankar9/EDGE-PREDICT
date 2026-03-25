"""Weighted ensemble that combines multiple model predictions.

Re-exports EnsemblePredictor from sharpedge.core.ensemble for backwards compatibility.
"""
from sharpedge.core.ensemble import EnsemblePredictor

__all__ = ["EnsemblePredictor"]
