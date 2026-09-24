def test_package_imports():
    from eval.label_probe import features, protocol
    assert features.UNIQUE_COMPS[:3] == (0, 1, 2)


def test_fixtures_shape(tiny_reps, tiny_bank):
    X, y = tiny_reps
    assert X.shape == (200, 24)
    assert y.shape == (200,)
    assert tiny_bank.shape == (40, 3, 10, 8)
