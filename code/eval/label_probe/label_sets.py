"""
Finds label-set directories under a labeled-data tree. Lives here (not in
runner.py) so both eval.runner (`--evals label_probe`) and
merge_label_sets.py (combining several sets into one) can share it without
a circular import -- runner.py imports label_probe modules, not the reverse.
"""
from __future__ import annotations

import os
from typing import Optional


def has_usable_labels(candidate_dir: str) -> bool:
    """A labels.tsv that exists but is empty (a dataset mid-conversion, or
    one where every row turned out NaN and got filtered out upstream) is not
    a usable label set -- callers should skip it rather than hand it to
    run_study, which would just raise once it finds zero spectra."""
    path = os.path.join(candidate_dir, "labels.tsv")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        return any(line.strip() for line in f)


def resolve_label_sets(labeled_data_dir: Optional[str]) -> dict:
    """labeled_data_dir is either ONE label set (it has labels.tsv directly)
    or a PARENT of several -- at ANY nesting depth, not just one level down,
    since real label trees group sets under campaign/site/whatever folders
    of their own. A label set is any directory with a non-empty labels.tsv;
    once one is found, its own subtree is not searched further (a label set
    is a leaf -- a labels.tsv nested inside another label set's directory is
    data for that set, e.g. under wav/, not a second set). A directory whose
    labels.tsv is empty (see has_usable_labels) is silently excluded, same
    as one with no labels.tsv at all -- both mean "nothing to probe here
    yet", and the walk keeps going past it in case a real set lives deeper.

    Returns {name: path}, name = the relative path from labeled_data_dir
    with '/' separators (e.g. "campaign1/site_A"), or just the folder's own
    basename for the single-set case. Empty dict if labeled_data_dir doesn't
    qualify as either (missing, or no usable labels.tsv anywhere under it)."""
    if not labeled_data_dir or not os.path.isdir(labeled_data_dir):
        return {}
    if has_usable_labels(labeled_data_dir):
        return {os.path.basename(labeled_data_dir.rstrip("/")) or labeled_data_dir: labeled_data_dir}
    sets = {}
    for root, dirnames, filenames in os.walk(labeled_data_dir):
        if "labels.tsv" in filenames and has_usable_labels(root):
            name = os.path.relpath(root, labeled_data_dir).replace(os.sep, "/")
            sets[name] = root
            dirnames[:] = []  # usable leaf found -- don't descend into it further
        else:
            dirnames.sort()  # deterministic traversal order
    return dict(sorted(sets.items()))
