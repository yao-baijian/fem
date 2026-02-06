"""
Test suite for ml_alpha module - ML-based parameter prediction.

Tests model training, prediction, and dataset handling for alpha parameter optimization.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from ml_alpha.model import create_default_model, save_model, load_model
from ml_alpha.dataset import FIELDNAMES, extract_features_from_placer
from ml_alpha.train import train_from_csv, PRE_ALPHA_FEATURES
from ml_alpha.predict import predict_alpha


class TestMLAlphaModel:
    """Test ml_alpha model creation, saving, and loading."""

    def test_create_default_model(self):
        """Test that default model is created with correct parameters."""
        model = create_default_model()
        assert model is not None
        assert hasattr(model, 'fit')
        assert hasattr(model, 'predict')
        assert model.n_estimators == 200
        assert model.max_depth == 8

    def test_create_model_with_custom_params(self):
        """Test model creation with custom parameters."""
        model = create_default_model(n_estimators=100, max_depth=5)
        assert model.n_estimators == 100
        assert model.max_depth == 5

    def test_save_and_load_model(self):
        """Test model persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Create and save model
            model = create_default_model()
            save_model(model, model_path)
            assert os.path.exists(model_path)

            # Load model
            loaded_model = load_model(model_path)
            assert loaded_model is not None
            assert loaded_model.n_estimators == model.n_estimators
            assert loaded_model.max_depth == model.max_depth

    def test_load_nonexistent_model(self):
        """Test loading non-existent model returns None."""
        model = load_model('/nonexistent/path/model.pkl')
        assert model is None


class TestMLAlphaDataset:
    """Test ml_alpha dataset utilities."""

    def test_fieldnames_completeness(self):
        """Test that FIELDNAMES contains all expected fields."""
        expected_fields = [
            "instance", "opti_insts_num", "avail_sites_num", "fixed_insts_num",
            "utilization", "logic_area_length", "logic_area_width", "io_height",
            "net_count", "hpwl_before", "hpwl_after", "overlap_after", "alpha"
        ]
        assert FIELDNAMES == expected_fields

    def test_pre_alpha_features(self):
        """Test PRE_ALPHA_FEATURES excludes target variables and identifiers."""
        assert "alpha" not in PRE_ALPHA_FEATURES
        assert "hpwl_after" not in PRE_ALPHA_FEATURES
        assert "overlap_after" not in PRE_ALPHA_FEATURES
        assert "instance" not in PRE_ALPHA_FEATURES  # instance is string identifier, not numeric
        assert "opti_insts_num" in PRE_ALPHA_FEATURES


class TestMLAlphaTraining:
    """Test ml_alpha training functionality."""

    def test_train_from_csv_with_valid_data(self):
        """Test training with valid CSV data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_data.csv')
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Create synthetic training data
            num_samples = 50
            data = {
                "instance": [f"test_{i}" for i in range(num_samples)],
                "opti_insts_num": np.random.randint(50, 200, num_samples),
                "avail_sites_num": np.random.randint(100, 300, num_samples),
                "fixed_insts_num": np.random.randint(10, 50, num_samples),
                "utilization": np.random.uniform(0.3, 0.9, num_samples),
                "logic_area_length": np.random.randint(10, 30, num_samples),
                "logic_area_width": np.random.randint(10, 30, num_samples),
                "io_height": np.random.randint(5, 15, num_samples),
                "net_count": np.random.randint(100, 500, num_samples),
                "hpwl_before": np.random.uniform(1000, 5000, num_samples),
                "hpwl_after": np.random.uniform(800, 4000, num_samples),
                "overlap_after": np.random.randint(0, 10, num_samples),
                "alpha": np.random.uniform(0.5, 2.0, num_samples)
            }

            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            # Monkey-patch MODEL_PATH for this test
            import ml_alpha.train
            original_model_path = ml_alpha.train.MODEL_PATH
            ml_alpha.train.MODEL_PATH = model_path

            try:
                result = train_from_csv(csv_path, target="alpha", test_size=0.2)

                assert "mse" in result
                assert "model_path" in result
                assert result["mse"] >= 0
                assert os.path.exists(model_path)
            finally:
                ml_alpha.train.MODEL_PATH = original_model_path

    def test_train_from_csv_missing_file(self):
        """Test training with non-existent CSV raises error."""
        with pytest.raises(FileNotFoundError):
            train_from_csv("/nonexistent/path.csv")

    def test_train_from_csv_missing_target(self):
        """Test training with missing target column raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, 'test_data.csv')

            # Create CSV without target column
            data = {
                "instance": ["test_1", "test_2"],
                "opti_insts_num": [100, 120]
            }
            df = pd.DataFrame(data)
            df.to_csv(csv_path, index=False)

            with pytest.raises(ValueError, match="target column not present"):
                train_from_csv(csv_path, target="alpha")


class TestMLAlphaPrediction:
    """Test ml_alpha prediction functionality."""

    def test_predict_alpha_without_model(self):
        """Test prediction without trained model raises error."""
        # Temporarily ensure no model exists
        import ml_alpha.predict
        from ml_alpha import model as model_module

        # Save original load_model function
        original_load = model_module.load_model

        # Mock load_model to return None
        model_module.load_model = lambda path=None: None
        ml_alpha.predict.load_model = lambda: None

        try:
            feature_row = {
                "instance": "test",
                "opti_insts_num": 100,
                "avail_sites_num": 200,
                "fixed_insts_num": 20,
                "utilization": 0.5,
                "logic_area_length": 20,
                "logic_area_width": 20,
                "io_height": 10,
                "net_count": 150,
                "hpwl_before": 2000.0
            }

            with pytest.raises(RuntimeError, match="No trained model found"):
                predict_alpha(feature_row)
        finally:
            # Restore original function
            model_module.load_model = original_load
            ml_alpha.predict.load_model = model_module.load_model

    def test_predict_alpha_with_trained_model(self):
        """Test prediction with a trained model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.pkl')

            # Train a simple model
            X = np.random.randn(50, len(PRE_ALPHA_FEATURES))
            y = np.random.uniform(0.5, 2.0, 50)

            model = create_default_model()
            model.fit(X, y)
            save_model(model, model_path)

            # Mock load_model to use our test model
            import ml_alpha.predict
            ml_alpha.predict.load_model = lambda: load_model(model_path)

            try:
                feature_row = {
                    "instance": "test",
                    "opti_insts_num": 100,
                    "avail_sites_num": 200,
                    "fixed_insts_num": 20,
                    "utilization": 0.5,
                    "logic_area_length": 20,
                    "logic_area_width": 20,
                    "io_height": 10,
                    "net_count": 150,
                    "hpwl_before": 2000.0
                }

                alpha = predict_alpha(feature_row)
                assert isinstance(alpha, float)
                assert alpha > 0  # Alpha should be positive
            finally:
                # Restore original load_model
                from ml_alpha import model as model_module
                ml_alpha.predict.load_model = model_module.load_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
