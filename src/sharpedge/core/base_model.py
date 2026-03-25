"""Abstract base predictor interface for all sport-specific models."""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class BasePredictor(ABC):
    """Abstract interface that ALL sport-specific models must implement."""

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BasePredictor":
        """Train the model on historical data."""

    @abstractmethod
    def predict_proba(self, *args, **kwargs) -> NDArray[np.float64]:
        """Return probability array for the primary market.

        Shape: (n_samples, n_outcomes).
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique model name for ensemble tracking."""
