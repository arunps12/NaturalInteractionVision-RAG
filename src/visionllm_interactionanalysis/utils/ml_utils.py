"""ML utilities: seeding, device selection, checkpointing."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch

from visionllm_interactionanalysis.utils.exception import PipelineException
from visionllm_interactionanalysis.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch for reproducibility."""
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Seed set to %d", seed)
    except Exception as e:
        raise PipelineException("Failed to set random seed", e)


def get_device(device_cfg: str = "auto") -> torch.device:
    """Resolve a torch device from config string ('auto', 'cuda', 'cpu')."""
    try:
        if device_cfg == "cpu":
            return torch.device("cpu")
        if device_cfg in ("auto", "cuda"):
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA unavailable — falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_cfg)
    except Exception as e:
        raise PipelineException(f"Failed to resolve device: {device_cfg}", e)


def collate_fn(batch: list) -> tuple:
    """Custom collate for detection: returns (list[Tensor], list[dict])."""
    return tuple(zip(*batch))


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model state dict + optional metadata."""
    try:
        payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
        if extra:
            payload.update(extra)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(payload, path)
        logger.info("Checkpoint saved: %s", path)
    except Exception as e:
        raise PipelineException(f"Failed to save checkpoint: {path}", e)


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load checkpoint and restore model state dict. Returns full payload."""
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Checkpoint loaded: %s", path)
        return ckpt
    except Exception as e:
        raise PipelineException(f"Failed to load checkpoint: {path}", e)
