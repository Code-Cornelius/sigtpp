"""Unit tests for the guarded recursive directory delete."""

import os

from src.utils.utils_file import delete_dir_tree_safe


def _make_tree(base, name, payload=b"x" * 10):
    directory = os.path.join(base, name)
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "f.bin"), "wb") as handle:
        handle.write(payload)
    return directory


def test_delete_dir_tree_safe_removes_dir_under_root_and_returns_bytes(tmp_path):
    root = os.path.join(str(tmp_path), "models")
    os.makedirs(root)
    target = _make_tree(root, "m1", payload=b"x" * 123)

    freed = delete_dir_tree_safe(target, must_be_under=root)

    assert not os.path.isdir(target)
    assert freed == 123


def test_delete_dir_tree_safe_refuses_path_outside_root(tmp_path):
    root = os.path.join(str(tmp_path), "models")
    os.makedirs(root)
    outside = _make_tree(str(tmp_path), "outside")

    freed = delete_dir_tree_safe(outside, must_be_under=root)

    assert os.path.isdir(outside)  # untouched
    assert freed == 0


def test_delete_dir_tree_safe_refuses_root_itself(tmp_path):
    root = os.path.join(str(tmp_path), "models")
    os.makedirs(root)

    freed = delete_dir_tree_safe(root, must_be_under=root)

    assert os.path.isdir(root)
    assert freed == 0


def test_delete_dir_tree_safe_missing_path_is_noop(tmp_path):
    root = os.path.join(str(tmp_path), "models")
    os.makedirs(root)

    freed = delete_dir_tree_safe(os.path.join(root, "does_not_exist"), must_be_under=root)

    assert freed == 0
