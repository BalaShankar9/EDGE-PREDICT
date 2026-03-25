"""Tests for FeatureGroup re-export and new location."""

from sharpedge.core.base_features import FeatureGroup as CoreFeatureGroup
from sharpedge.ml.features.base import FeatureGroup as OldFeatureGroup


class TestFeatureGroupReExport:
    def test_new_import_works(self):
        assert CoreFeatureGroup is not None

    def test_old_import_works(self):
        assert OldFeatureGroup is not None

    def test_same_class(self):
        assert CoreFeatureGroup is OldFeatureGroup
