"""
Unit and regression test for the cyclops package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import cyclops


def test_cyclops_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "cyclops" in sys.modules