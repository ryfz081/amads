def test_melsim_import():
    """Test that melsim can be imported."""
    from amads.melody.similarity.melsim import compute_similarity

    assert callable(compute_similarity)
