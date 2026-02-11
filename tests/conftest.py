"""Test fixtures for VisionLLM InteractionAnalysis."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that auto-cleans."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_coco_json(tmp_dir: str) -> str:
    """Create a minimal valid COCO annotation file."""
    coco = {
        "images": [
            {"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img002.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 100, 100], "area": 10000, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 20, 50, 50], "area": 2500, "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "person"},
        ],
    }
    path = os.path.join(tmp_dir, "instances.json")
    Path(path).write_text(json.dumps(coco), encoding="utf-8")
    return path


@pytest.fixture
def sample_manifest(tmp_dir: str, sample_coco_json: str) -> str:
    """Create a minimal ingestion manifest YAML."""
    import yaml

    # Create dummy image files
    img_dir = os.path.join(tmp_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    for name in ["img001.jpg", "img002.jpg"]:
        Path(os.path.join(img_dir, name)).write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header

    manifest = {
        "dataset_name": "test-coco",
        "dataset_format": "raw",
        "created_at": "2026-01-01T00:00:00",
        "ingestion": {"mode": "register"},
        "paths": {
            "train_images": img_dir,
            "val_images": img_dir,
        },
        "annotations": {
            "train": sample_coco_json,
            "val": sample_coco_json,
        },
    }
    path = os.path.join(tmp_dir, "manifest.yaml")
    Path(path).write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path
