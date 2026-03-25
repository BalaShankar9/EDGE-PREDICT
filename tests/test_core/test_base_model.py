"""Tests for abstract BasePredictor interface."""

import numpy as np
import pytest

from sharpedge.core.base_model import BasePredictor


class TestBasePredictor:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            BasePredictor()

    def test_concrete_subclass_works(self):
        class MyModel(BasePredictor):
            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return np.array([[0.5, 0.3, 0.2]])

            @property
            def name(self):
                return "my_model"

        model = MyModel()
        assert model.name == "my_model"
        proba = model.predict_proba(None)
        assert proba.shape == (1, 3)
        assert model.fit(None, None) is model

    def test_partial_implementation_raises(self):
        class IncompleteModel(BasePredictor):
            def fit(self, X, y):
                return self

            # missing predict_proba and name

        with pytest.raises(TypeError):
            IncompleteModel()
