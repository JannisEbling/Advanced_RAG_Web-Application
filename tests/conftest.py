"""
This file contains pytest fixtures that can be shared across multiple test files.
Fixtures defined here are available to all test files without explicit imports.
"""

import pytest
import os
import sys
from pathlib import Path

# Add the src directory to Python path for importing project modules
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

@pytest.fixture(scope="session")
def test_data_path():
    """Fixture to provide path to test data directory."""
    return Path(__file__).parent / "data"

@pytest.fixture(scope="session")
def model_artifacts_path():
    """Fixture to provide path to test model artifacts."""
    return Path(__file__).parent / "models"

@pytest.fixture(scope="session")
def sample_config():
    """Fixture to provide sample configuration for testing."""
    return {
        "model_params": {
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 1
        },
        "data_params": {
            "train_split": 0.8,
            "validation_split": 0.1,
            "test_split": 0.1
        }
    }

@pytest.fixture(autouse=True)
def env_setup():
    """Automatically set up test environment variables."""
    os.environ["ENVIRONMENT"] = "test"
    yield
    # Clean up after tests
    if "ENVIRONMENT" in os.environ:
        del os.environ["ENVIRONMENT"]
