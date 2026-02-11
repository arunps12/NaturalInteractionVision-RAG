"""File I/O helpers: YAML, JSON, directory creation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml

from visionllm_interactionanalysis.utils.exception import PipelineException
from visionllm_interactionanalysis.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_dir(dir_path: str | Path) -> None:
    """Create directory tree (idempotent)."""
    try:
        os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        raise PipelineException(f"Failed to create directory: {dir_path}", e)


# ── YAML ─────────────────────────────────────────────────────────────
def read_yaml(file_path: str | Path) -> dict[str, Any]:
    """Read a YAML file and return a dict."""
    try:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        content = yaml.safe_load(p.read_text(encoding="utf-8"))
        if content is None:
            raise ValueError(f"YAML file is empty: {file_path}")
        logger.debug("YAML read: %s", file_path)
        return content
    except Exception as e:
        raise PipelineException(f"Failed to read YAML: {file_path}", e)


def write_yaml(file_path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict to a YAML file (creates parent dirs)."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.debug("YAML written: %s", file_path)
    except Exception as e:
        raise PipelineException(f"Failed to write YAML: {file_path}", e)


# ── JSON ─────────────────────────────────────────────────────────────
def load_json(file_path: str | Path) -> dict[str, Any]:
    """Read a JSON file and return a dict."""
    try:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if data is None:
            raise ValueError(f"JSON file is empty: {file_path}")
        logger.debug("JSON read: %s", file_path)
        return data
    except Exception as e:
        raise PipelineException(f"Failed to read JSON: {file_path}", e)


def write_json(file_path: str | Path, data: dict[str, Any]) -> None:
    """Write a dict to a JSON file (creates parent dirs)."""
    try:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        logger.debug("JSON written: %s", file_path)
    except Exception as e:
        raise PipelineException(f"Failed to write JSON: {file_path}", e)
