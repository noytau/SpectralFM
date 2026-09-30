"""Shared fixtures. Deliberately tiny and synthetic -- the real 640MB rep
cache lives in another clone and must never be a unit-test dependency."""
import numpy as np
import pytest


@pytest.fixture
def tiny_reps():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 24)).astype(np.float32)
    y = (2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * X[:, 2]
         + 0.1 * rng.normal(size=200)).astype(np.float64)
    return X, y


@pytest.fixture
def tiny_bank():
    rng = np.random.default_rng(1)
    return rng.normal(size=(40, 3, 10, 8)).astype(np.float32)
