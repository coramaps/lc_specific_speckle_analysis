"""Basic tests for the package."""

import unittest
from src.lc_speckle_analysis import __version__


class TestPackage(unittest.TestCase):
    """Test basic package functionality."""
    
    def test_version(self):
        """Test version is set."""
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
