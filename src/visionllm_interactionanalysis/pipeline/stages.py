"""DVC-compatible per-stage CLI runners.

Each stage is a standalone command that:
  1. Reads its upstream artifact(s) from JSON marker files.
  2. Runs the stage logic.
  3. Writes its own artifact to a JSON marker file.

DVC calls these commands and tracks deps / outs / params so that
unchanged stages are skipped automatically on ``dvc repro``.

Usage (called by DVC, not directly):
    python -m visionllm_interactionanalysis.pipeline.stages data_ingestion
    python -m visionllm_interactionanalysis.pipeline.stages data_validation
    python -m visionllm_interactionanalysis.pipeline.stages data_transformation
    python -m visionllm_interactionanalysis.pipeline.stages model_training
"""

from __future__ import annotations

import argparse
import json
import os

from visionllm_interactionanalysis.config import constants as C
from visionllm_interactionanalysis.config.artifacts import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from visionllm_interactionanalysis.config.settings import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    PipelineConfig,
)
from visionllm_interactionanalysis.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────

_DVC_DIR = C.DVC_RUN_DIR


def _artifact_path(stage_name: str) -> str:
    """Return the canonical path for a stage's artifact marker."""
    return os.path.join(_DVC_DIR, stage_name, "artifact.json")


def _write_artifact(stage_name: str, artifact_dict: dict) -> None:
    path = _artifact_path(stage_name)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact_dict, f, indent=2)
    logger.info("Artifact marker → %s", path)


def _read_artifact(stage_name: str) -> dict:
    path = _artifact_path(stage_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Upstream artifact not found: {path}. "
            f"Run the '{stage_name}' stage first (dvc repro {stage_name})."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pipeline() -> PipelineConfig:
    """Create a PipelineConfig with a fixed DVC run directory."""
    return PipelineConfig(run_dir=_DVC_DIR)


# ── Stage runners ────────────────────────────────────────────────────

def run_data_ingestion() -> None:
    """Stage 1: Data Ingestion."""
    from visionllm_interactionanalysis.detection.data_ingestion import DataIngestion

    pipeline = _pipeline()
    cfg = DataIngestionConfig.from_pipeline(pipeline)
    artifact = DataIngestion(cfg).run()
    _write_artifact("data_ingestion", artifact.model_dump())


def run_data_validation() -> None:
    """Stage 2: Data Validation."""
    from visionllm_interactionanalysis.detection.data_validation import DataValidation

    pipeline = _pipeline()
    ing_data = _read_artifact("data_ingestion")
    ing_artifact = DataIngestionArtifact(**ing_data)

    cfg = DataValidationConfig.from_pipeline(pipeline)
    artifact = DataValidation(cfg, ing_artifact).run()
    _write_artifact("data_validation", artifact.model_dump())


def run_data_transformation() -> None:
    """Stage 3: Data Transformation."""
    from visionllm_interactionanalysis.detection.data_transformation import DataTransformation

    pipeline = _pipeline()
    ing_artifact = DataIngestionArtifact(**_read_artifact("data_ingestion"))
    val_artifact = DataValidationArtifact(**_read_artifact("data_validation"))

    cfg = DataTransformationConfig.from_pipeline(pipeline)
    artifact = DataTransformation(cfg, ing_artifact, val_artifact).run()
    _write_artifact("data_transformation", artifact.model_dump())


def run_model_training() -> None:
    """Stage 4: Model Training."""
    from visionllm_interactionanalysis.detection.model_trainer import ModelTrainer

    pipeline = _pipeline()
    tx_artifact = DataTransformationArtifact(**_read_artifact("data_transformation"))

    cfg = ModelTrainerConfig.from_pipeline(pipeline)
    artifact = ModelTrainer(cfg, tx_artifact).run()
    _write_artifact("model_training", artifact.model_dump())


# ── CLI dispatch ─────────────────────────────────────────────────────

_STAGES = {
    "data_ingestion": run_data_ingestion,
    "data_validation": run_data_validation,
    "data_transformation": run_data_transformation,
    "model_training": run_model_training,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single pipeline stage (designed for DVC).",
    )
    parser.add_argument(
        "stage",
        choices=list(_STAGES.keys()),
        help="Stage to execute.",
    )
    args = parser.parse_args()

    logger.info("=== DVC Stage: %s ===", args.stage)
    _STAGES[args.stage]()
    logger.info("=== DVC Stage %s Complete ===", args.stage)


if __name__ == "__main__":
    main()
