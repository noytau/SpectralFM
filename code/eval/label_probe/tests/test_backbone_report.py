import os

import numpy as np

from .. import backbone_metrics as bm
from ..backbone_report import JS, Report
from ..nested import run_nested_for_dir


def _fake_run(path, raw, y, rng):
    os.makedirs(path, exist_ok=True)
    n = len(y)
    bank = {f"bank__layer{i}": rng.normal(size=(n, 1, 10, 6)).astype(np.float32) for i in (0, 1)}
    np.savez(os.path.join(path, "bank.npz"), input_raw=raw, input_z=raw, y=y,
             _meta=np.array([repr({"n": n})]), **bank)
    run_nested_for_dir(path)


def test_report_renders_from_nested_runs(tmp_path):
    rng = np.random.default_rng(0)
    backbones = bm.BACKBONES[:2]
    # the same spectra and labels under every backbone, as in a real sweep
    data = {}
    for ds in ("dataset0001", bm.MERGED, bm.LD):
        n = 30
        raw = rng.normal(size=(n, 1, 12)).astype(np.float32)
        data[ds] = (raw, raw[:, 0] @ rng.normal(size=12))
    for b in backbones:
        for ds, (raw, y) in data.items():
            _fake_run(bm._run_dir(str(tmp_path), b["tag"], ds), raw, y, rng)
    M = bm.collect(str(tmp_path), backbones=backbones)
    assert M["label_sets"] == ["dataset0001"]
    assert M["ranking"][bm.LD]["order"] and len(M["ranking"][bm.LD]["gaps"]) == 1
    page = Report(M, {}).render("", JS)
    assert "<title>" in page and "dataset0001" in page and "paired SD" in page
