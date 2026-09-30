"""
Aggregate step-5b shards into the extended label-efficiency report:
faceted curves with uncertainty bands, a labels-to-ceiling table, and a
TabPFN true-vs-predicted panel against the actual labels.
"""
from __future__ import annotations

import glob, json, os
import numpy as np

STAGES = ("input_raw", "input_z", "fe", "extract_features", "proj", "transformer")
SLABEL = {"input_raw":"Raw input","input_z":"Input (z-scored)","fe":"Post-FE",
          "extract_features":"Post-FE LN","proj":"Post-projection","transformer":"Post-transformer"}


def load(out_dir):
    cells = {}
    for p in sorted(glob.glob(os.path.join(out_dir, "step5b_*.json"))):
        d = json.load(open(p))
        for k, v in d["cells"].items():
            cells[k] = v
    return cells


def best_at(cells, n_comp, stage, n_train, exclude=("dummy",)):
    """Best probe for a cell by median R2, and its full stats."""
    best, br = None, -np.inf
    for k, v in cells.items():
        p = k.split("|")
        if len(p) != 4: continue
        c, s, probe, nt = p
        if (int(c), s, int(nt)) != (n_comp, stage, n_train) or probe in exclude:
            continue
        if v["r2_median"] > br:
            br, best = v["r2_median"], (probe, v)
    return best


def labels_to_reach(cells, n_comp, stage, target, ns):
    """Smallest n_train whose median R2 reaches `target` (interpolated)."""
    prev = None
    for nt in ns:
        g = best_at(cells, n_comp, stage, nt)
        if not g: continue
        v = g[1]["r2_median"]
        if v >= target:
            if prev and prev[1] < target:
                n0, v0 = prev
                f = (target - v0) / (v - v0) if v != v0 else 0
                return int(round(n0 * (nt / n0) ** f))
            return nt
        prev = (nt, v)
    return None
