"""Tests for melsim integration."""


def test_melsim_import():
    """Test that melsim can be imported."""
    from amads.melody.similarity.melsim import get_similarity

    assert callable(get_similarity)
