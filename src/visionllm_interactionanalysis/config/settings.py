"""Pydantic-validated pipeline stage configurations."""

from __future__ import annotations

import os
from datetime import datetime

from pydantic import BaseModel, Field

from visionllm_interactionanalysis.config import constants as C


# ── Pipeline root ────────────────────────────────────────────────────
class PipelineConfig(BaseModel):
    """Top-level config that stamps a run with a timestamp.

    For DVC mode (fixed output paths), pass ``run_dir`` directly
    to skip the timestamp-based directory.
    """

    pipeline_name: str = C.PIPELINE_NAME
    artifacts_dir: str = C.ARTIFACTS_DIR
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))
    run_dir: str | None = None  # When set, overrides artifacts_dir/timestamp

    @property
    def resolved_run_dir(self) -> str:
        """Return the effective run directory."""
        if self.run_dir is not None:
            return self.run_dir
        return os.path.join(self.artifacts_dir, self.timestamp)


# ── Stage 1 ──────────────────────────────────────────────────────────
class DataIngestionConfig(BaseModel):
    """Stage-1 config: where raw data lives, where manifest is written."""

    dataset_name: str = C.DATASET_NAME
    ingestion_mode: str = C.INGESTION_MODE
    dataset_format: str = "raw"

    # Raw data paths (configurable via env)
    raw_data_dir: str = C.RAW_DATA_DIR
    raw_train_image_dir: str = C.RAW_TRAIN_IMAGE_DIR
    raw_val_image_dir: str = C.RAW_VAL_IMAGE_DIR
    raw_annotation_dir: str = C.RAW_ANNOTATION_DIR
    raw_train_annotation_file: str = C.RAW_TRAIN_ANN_FILE
    raw_val_annotation_file: str = C.RAW_VAL_ANN_FILE

    # Artifact output dir (set by pipeline)
    stage_dir: str = ""
    manifest_file: str = ""

    @classmethod
    def from_pipeline(cls, pipeline: PipelineConfig, **overrides) -> DataIngestionConfig:
        stage_dir = os.path.join(pipeline.resolved_run_dir, C.DATA_INGESTION_DIR)
        return cls(
            stage_dir=stage_dir,
            manifest_file=os.path.join(stage_dir, C.DATA_INGESTION_MANIFEST),
            **overrides,
        )


# ── Stage 2 ──────────────────────────────────────────────────────────
class DataValidationConfig(BaseModel):
    """Stage-2 config: validation report location + policy knobs."""

    schema_file: str = C.SCHEMA_FILE
    require_val: bool = True
    drop_invalid_bbox: bool = C.DROP_INVALID_BBOX
    max_invalid_bbox_allowed: int = C.MAX_INVALID_BBOX_ALLOWED

    stage_dir: str = ""
    report_file: str = ""

    @classmethod
    def from_pipeline(cls, pipeline: PipelineConfig, **overrides) -> DataValidationConfig:
        stage_dir = os.path.join(pipeline.resolved_run_dir, C.DATA_VALIDATION_DIR)
        return cls(
            stage_dir=stage_dir,
            report_file=os.path.join(stage_dir, C.VALIDATION_REPORT),
            **overrides,
        )


# ── Stage 3 ──────────────────────────────────────────────────────────
class DataTransformationConfig(BaseModel):
    """Stage-3 config: cleaned annotation output paths."""

    stage_dir: str = ""
    cleaned_annotation_dir: str = ""
    cleaned_train_ann_file: str = ""
    cleaned_val_ann_file: str = ""
    training_manifest_file: str = ""

    @classmethod
    def from_pipeline(cls, pipeline: PipelineConfig, **overrides) -> DataTransformationConfig:
        stage_dir = os.path.join(pipeline.resolved_run_dir, C.DATA_TRANSFORMATION_DIR)
        ann_dir = os.path.join(stage_dir, "annotations")
        return cls(
            stage_dir=stage_dir,
            cleaned_annotation_dir=ann_dir,
            cleaned_train_ann_file=os.path.join(ann_dir, C.CLEANED_TRAIN_ANN),
            cleaned_val_ann_file=os.path.join(ann_dir, C.CLEANED_VAL_ANN),
            training_manifest_file=os.path.join(stage_dir, C.TRAINING_MANIFEST),
            **overrides,
        )


# ── Stage 4 ──────────────────────────────────────────────────────────
class ModelTrainerConfig(BaseModel):
    """Stage-4 config: model output paths."""

    model_config_file: str = C.MODEL_CONFIG_FILE

    stage_dir: str = ""
    report_file: str = ""
    best_model_file: str = ""
    last_model_file: str = ""

    @classmethod
    def from_pipeline(cls, pipeline: PipelineConfig, **overrides) -> ModelTrainerConfig:
        stage_dir = os.path.join(pipeline.resolved_run_dir, C.MODEL_TRAINER_DIR)
        return cls(
            stage_dir=stage_dir,
            report_file=os.path.join(stage_dir, C.TRAINING_REPORT),
            best_model_file=os.path.join(stage_dir, C.BEST_MODEL_FILE),
            last_model_file=os.path.join(stage_dir, C.LAST_MODEL_FILE),
            **overrides,
        )

    model_config = {"protected_namespaces": ()}
