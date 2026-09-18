"""resolve_label_sets is pure filesystem logic (no model/GPU), so it gets
its own fast test module rather than needing the label_probe test fixtures."""
from __future__ import annotations

import os

from eval.runner import resolve_label_sets


def _touch(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


def test_missing_or_none_dir_returns_empty(tmp_path):
    assert resolve_label_sets(None) == {}
    assert resolve_label_sets(str(tmp_path / "does_not_exist")) == {}


def test_dir_with_no_labels_tsv_anywhere_returns_empty(tmp_path):
    (tmp_path / "wav").mkdir()
    _touch(str(tmp_path / "wav" / "spec_0.wav"))
    assert resolve_label_sets(str(tmp_path)) == {}


def test_single_label_set_at_top_level(tmp_path):
    _touch(str(tmp_path / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {tmp_path.name: str(tmp_path)}


def test_one_level_of_subfolders(tmp_path):
    _touch(str(tmp_path / "A" / "labels.tsv"))
    _touch(str(tmp_path / "B" / "labels.tsv"))
    (tmp_path / "C_no_labels").mkdir()
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "A": str(tmp_path / "A"),
        "B": str(tmp_path / "B"),
    }


def test_deeply_nested_label_sets(tmp_path):
    """Real label trees group sets under campaign/site folders of their own
    -- not just one level down."""
    _touch(str(tmp_path / "campaign1" / "site_A" / "labels.tsv"))
    _touch(str(tmp_path / "campaign1" / "site_B" / "labels.tsv"))
    _touch(str(tmp_path / "campaign2" / "sub" / "site_C" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "campaign1/site_A": str(tmp_path / "campaign1" / "site_A"),
        "campaign1/site_B": str(tmp_path / "campaign1" / "site_B"),
        "campaign2/sub/site_C": str(tmp_path / "campaign2" / "sub" / "site_C"),
    }


def test_mixed_depth_label_sets(tmp_path):
    """A set directly under the parent alongside sets several levels deeper
    -- both must be found."""
    _touch(str(tmp_path / "direct_set" / "labels.tsv"))
    _touch(str(tmp_path / "group" / "nested_set" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {
        "direct_set": str(tmp_path / "direct_set"),
        "group/nested_set": str(tmp_path / "group" / "nested_set"),
    }


def test_a_label_sets_own_subtree_is_not_searched_further(tmp_path):
    """A labels.tsv nested inside another label set's directory (e.g. an
    oddly-placed file under its own wav/) belongs to that set, not a second
    one -- once a label set is found, its subtree is a leaf."""
    _touch(str(tmp_path / "outer" / "labels.tsv"))
    _touch(str(tmp_path / "outer" / "wav" / "labels.tsv"))
    sets = resolve_label_sets(str(tmp_path))
    assert sets == {"outer": str(tmp_path / "outer")}
