import os

import numpy as np
import pytest

from eval.label_probe import readouts as ro


def test_pool_stats_and_poolings_are_consistent():
    assert len(ro.POOL_STATS) == 10
    assert len(ro.BANK_STAGES) == 15
    assert ro.BANK_STAGES[:2] == ("fe", "extract_features")
    assert ro.BANK_STAGES[2] == "layer0" and ro.BANK_STAGES[-1] == "layer12"
    for name, stats in ro.POOLINGS.items():
        assert len(stats) > 0, name
        for s in stats:
            assert s in ro.POOL_STATS, f"{name} references unknown stat {s}"


def test_build_readout_shape_and_order(tiny_bank):
    # 40 spectra, 3 comps, 10 stats, 8 channels
    X = ro.build_readout(tiny_bank, [0, 2], "mean_std")
    assert X.shape == (40, 2 * 2 * 8)   # 2 comps x 2 stats x 8 channels
    assert X.dtype == np.float32
    # component-major, then stat, then channel: first block is comp0/mean
    mean_i = ro.POOL_STATS.index("mean")
    np.testing.assert_allclose(X[:, :8], tiny_bank[:, 0, mean_i, :], rtol=1e-6)
    std_i = ro.POOL_STATS.index("std")
    np.testing.assert_allclose(X[:, 8:16], tiny_bank[:, 0, std_i, :], rtol=1e-6)
    # second component block starts at 16 and is comp *2*, not comp 1
    np.testing.assert_allclose(X[:, 16:24], tiny_bank[:, 2, mean_i, :], rtol=1e-6)


def test_build_readout_mean_matches_single_stat(tiny_bank):
    X = ro.build_readout(tiny_bank, [1], "mean")
    assert X.shape == (40, 8)
    np.testing.assert_allclose(X, tiny_bank[:, 1, ro.POOL_STATS.index("mean"), :],
                               rtol=1e-6)


def test_build_readout_rejects_unknown_pooling(tiny_bank):
    with pytest.raises(ValueError, match="unknown pooling"):
        ro.build_readout(tiny_bank, [0], "no_such_pooling")


def test_raw_moments_values():
    sig = np.zeros((2, 1, 8), dtype=np.float32)
    sig[0, 0] = np.arange(8)          # 0..7
    sig[1, 0] = np.full(8, 3.0)
    m = ro.raw_moments(sig)
    assert m.shape == (2, 1, 10)
    i = {s: k for k, s in enumerate(ro.POOL_STATS)}
    assert m[0, 0, i["mean"]] == pytest.approx(3.5)
    assert m[0, 0, i["max"]] == pytest.approx(7.0)
    assert m[0, 0, i["min"]] == pytest.approx(0.0)
    assert m[0, 0, i["first"]] == pytest.approx(0.0)
    assert m[0, 0, i["last"]] == pytest.approx(7.0)
    assert m[0, 0, i["seg0"]] == pytest.approx(0.5)    # mean of [0,1]
    assert m[0, 0, i["seg3"]] == pytest.approx(6.5)    # mean of [6,7]
    assert m[1, 0, i["std"]] == pytest.approx(0.0)


class _FakeHidden:
    """Stands in for a HF Data2VecAudioModel output with hidden_states."""

    def __init__(self, hidden_states):
        self.hidden_states = hidden_states
        self.last_hidden_state = hidden_states[-1]


class _FakeModel:
    """Minimal stand-in exposing exactly the surface extract_bank uses:
    feature_extractor -> [B, 512, T], feature_projection.layer_norm, and
    __call__(output_hidden_states=True) -> 13 x [B, T, 768]."""

    def __init__(self, T=7, n_layers=13):
        import torch
        self.T, self.n_layers, self.torch = T, n_layers, torch

        class _FE:
            def __call__(_s, x):
                b = x.shape[0]
                return torch.arange(b * 512 * T, dtype=torch.float32).reshape(b, 512, T)

        class _Proj:
            layer_norm = staticmethod(lambda t: t * 2.0)

        self.feature_extractor = _FE()
        self.feature_projection = _Proj()

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, input_values=None, output_hidden_states=False):
        b = input_values.shape[0]
        hs = [self.torch.full((b, self.T, 768), float(i)) for i in range(self.n_layers)]
        return _FakeHidden(hs)


def test_extract_bank_shapes_and_stats():
    import numpy as np
    from eval.label_probe import readouts as ro

    m = _FakeModel(T=7)
    sig = np.zeros((5, 245), dtype=np.float32)
    bank = ro.extract_bank(m, sig, device="cpu", batch_size=2)

    assert set(bank) == set(ro.BANK_STAGES)
    assert bank["fe"].shape == (5, 10, 512)
    assert bank["extract_features"].shape == (5, 10, 512)
    assert bank["layer0"].shape == (5, 10, 768)
    assert bank["layer12"].shape == (5, 10, 768)
    assert bank["layer0"].dtype == np.float16

    # layer i is the constant i, so every statistic equals i and std is 0
    i_mean = ro.POOL_STATS.index("mean")
    i_std = ro.POOL_STATS.index("std")
    assert np.allclose(bank["layer5"][:, i_mean, :], 5.0)
    assert np.allclose(bank["layer5"][:, i_std, :], 0.0)
    # extract_features is the layer-normed (x2) view of fe, so its mean is
    # exactly twice fe's
    np.testing.assert_allclose(
        bank["extract_features"][:, i_mean, :].astype(np.float32),
        2.0 * bank["fe"][:, i_mean, :].astype(np.float32), rtol=1e-2)


def test_build_bank_cache_atomic_write_and_round_trip(tmp_path, monkeypatch):
    """Covers the crash-safety fix: build_bank_cache must write via a temp
    file + os.replace so a successful run never leaves bank.npz.tmp behind,
    and load_bank_cache must round-trip the result. Fully offline: no
    checkpoint, no GPU -- the model and data loader are faked."""
    import numpy as np
    from eval.label_probe import readouts as ro
    import eval.checkpoint_loader as ckpt_mod
    import eval.data_loader as data_mod

    n, k, L = 3, 2, 245
    fake_raw = np.zeros((n, k, L), dtype=np.float32)
    fake_y = np.arange(n, dtype=np.float64)

    def _fake_load_labeled_data(labeled_data_dir, max_samples, seed, comps):
        return fake_raw, fake_y

    monkeypatch.setattr(data_mod, "load_labeled_data", _fake_load_labeled_data)
    monkeypatch.setattr(ckpt_mod.CheckpointLoader, "from_file",
                        staticmethod(lambda path: _FakeModel(T=4)))

    out_dir = str(tmp_path)
    path = ro.build_bank_cache(
        checkpoint_path="unused", labeled_data_dir="unused", out_dir=out_dir,
        comps=(0, 1), max_samples=n, device="cpu", batch_size=4, seed=0)

    assert path == os.path.join(out_dir, "bank.npz")
    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp"), "temp file left behind after a successful write"

    bank, input_raw, input_z, y, meta = ro.load_bank_cache(path)
    assert set(bank) == set(ro.BANK_STAGES)
    for stage in ro.BANK_STAGES:
        assert bank[stage].shape[:3] == (n, k, len(ro.POOL_STATS))
    assert input_raw.shape == (n, k, L)
    assert input_z.shape == (n, k, L)
    assert y.shape == (n,)
    assert meta["n"] == n
    assert meta["stages"] == ro.BANK_STAGES

    # idempotency: a second call must short-circuit on the existing file
    # rather than raising or re-invoking the (would-be-broken) loader.
    monkeypatch.setattr(ckpt_mod.CheckpointLoader, "from_file",
                        staticmethod(lambda path: (_ for _ in ()).throw(
                            AssertionError("should not rebuild an existing cache"))))
    path2 = ro.build_bank_cache(
        checkpoint_path="unused", labeled_data_dir="unused", out_dir=out_dir,
        comps=(0, 1), max_samples=n, device="cpu", batch_size=4, seed=0)
    assert path2 == path
