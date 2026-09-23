"""resolve_label_sets is pure filesystem logic (no model/GPU), so it gets
its own fast test module rather than needing the label_probe test fixtures.
Tests the shared implementation directly (eval.label_probe.label_sets);
eval.runner re-exports the same function."""
from __future__ import annotations

import os

from eval.label_probe.label_sets import resolve_label_sets


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def _write_labels(path: str, n_rows: int = 1) -> None:
    """A labels.tsv with real (fake) rows -- what a usable label set has."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for i in range(n_rows):
            f.write(f"dataset0000_comp0_spec_{i}.wav\t0.5\n")


def test_missing_or_none_dir_returns_empty(tmp_path):
    assert resolve_label_sets(None) == {}
    assert resolve_label_sets(str(tmp_path / "does_not_exist")) == {}


def test_dir_with_no_labels_tsv_anywhere_returns_empty(tmp_path):
    (tmp_path / "wav").mkdir()
    _touch(str(tmp_path / "wav" / "spec_0.wav"))
    assert resolve_label_sets(str(tmp_path)) == {}


def test_single_label_set_at_top_level(tmp_path):
    _write_labels(str(tmp_path / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {tmp_path.name: str(tmp_path)}


def test_one_level_of_subfolders(tmp_path):
    _write_labels(str(tmp_path / "A" / "labels.tsv"))
    _write_labels(str(tmp_path / "B" / "labels.tsv"))
    (tmp_path / "C_no_labels").mkdir()
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "A": str(tmp_path / "A"),
        "B": str(tmp_path / "B"),
    }


def test_deeply_nested_label_sets(tmp_path):
    """Real label trees group sets under campaign/site folders of their own
    -- not just one level down."""
    _write_labels(str(tmp_path / "campaign1" / "site_A" / "labels.tsv"))
    _write_labels(str(tmp_path / "campaign1" / "site_B" / "labels.tsv"))
    _write_labels(str(tmp_path / "campaign2" / "sub" / "site_C" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "campaign1/site_A": str(tmp_path / "campaign1" / "site_A"),
        "campaign1/site_B": str(tmp_path / "campaign1" / "site_B"),
        "campaign2/sub/site_C": str(tmp_path / "campaign2" / "sub" / "site_C"),
    }


def test_mixed_depth_label_sets(tmp_path):
    """A set directly under the parent alongside sets several levels deeper
    -- both must be found."""
    _write_labels(str(tmp_path / "direct_set" / "labels.tsv"))
    _write_labels(str(tmp_path / "group" / "nested_set" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "direct_set": str(tmp_path / "direct_set"),
        "group/nested_set": str(tmp_path / "group" / "nested_set"),
    }


def test_a_label_sets_own_subtree_is_not_searched_further(tmp_path):
    """A labels.tsv nested inside another label set's directory (e.g. an
    oddly-placed file under its own wav/) belongs to that set, not a second
    one -- once a label set is found, its subtree is a leaf."""
    _write_labels(str(tmp_path / "outer" / "labels.tsv"))
    _write_labels(str(tmp_path / "outer" / "wav" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {"outer": str(tmp_path / "outer")}


def test_empty_labels_tsv_at_top_level_is_not_a_set(tmp_path):
    """A dataset mid-conversion, or one where every row was NaN and got
    filtered upstream -- present but empty, not usable."""
    _touch(str(tmp_path / "labels.tsv"))
    assert resolve_label_sets(str(tmp_path)) == {}


def test_empty_labels_tsv_subfolder_is_skipped_but_siblings_are_found(tmp_path):
    _touch(str(tmp_path / "empty_one" / "labels.tsv"))
    _write_labels(str(tmp_path / "real_one" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {"real_one": str(tmp_path / "real_one")}


def test_empty_labels_tsv_does_not_block_a_real_set_deeper_in_the_same_branch(tmp_path):
    """An empty labels.tsv sitting in an intermediate directory must not
    prune the walk -- only a USABLE one is a leaf."""
    _touch(str(tmp_path / "group" / "labels.tsv"))
    _write_labels(str(tmp_path / "group" / "real_subset" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {"group/real_subset": str(tmp_path / "group" / "real_subset")}
