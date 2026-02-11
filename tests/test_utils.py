"""Tests for IO utilities (YAML, JSON)."""

import os
from pathlib import Path

from visionllm_interactionanalysis.utils import load_json, read_yaml, write_json, write_yaml


def test_yaml_round_trip(tmp_dir: str):
    path = os.path.join(tmp_dir, "test.yaml")
    data = {"key": "value", "nested": {"a": 1}}
    write_yaml(path, data)
    result = read_yaml(path)
    assert result == data


def test_json_round_trip(tmp_dir: str):
    path = os.path.join(tmp_dir, "test.json")
    data = {"items": [1, 2, 3], "ok": True}
    write_json(path, data)
    result = load_json(path)
    assert result == data


def test_write_creates_parents(tmp_dir: str):
    path = os.path.join(tmp_dir, "a", "b", "c", "deep.yaml")
    write_yaml(path, {"hello": "world"})
    assert Path(path).exists()
