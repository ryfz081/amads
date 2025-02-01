"""Tests for melsim integration."""

from tests.utils_test import only_on_ci_job


@only_on_ci_job("tests-melsim")
def test_melsim_import():
    """Test that melsim can be imported."""
    from amads.melody.similarity.melsim import get_similarity

    assert callable(get_similarity)
