"""Test configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    # Add sample data fixtures here
    pass
