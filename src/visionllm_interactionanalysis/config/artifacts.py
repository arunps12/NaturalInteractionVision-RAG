"""Pydantic models for stage artifacts (outputs)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class DataIngestionArtifact(BaseModel):
    """Produced by Stage 1."""

    stage_dir: str
    manifest_file: str
    ingestion_mode: str
    dataset_format: str


class DataValidationArtifact(BaseModel):
    """Produced by Stage 2."""

    stage_dir: str
    report_file: str
    validated: bool


class DataTransformationArtifact(BaseModel):
    """Produced by Stage 3."""

    stage_dir: str
    cleaned_train_ann_file: str
    cleaned_val_ann_file: str
    training_manifest_file: str
    dropped_train_count: int
    dropped_val_count: int


class ModelTrainerArtifact(BaseModel):
    """Produced by Stage 4."""

    stage_dir: str
    best_model_path: str
    last_model_path: str
    training_report_path: str
    best_metric_name: str
    best_metric_value: float
    hpo_enabled: bool
    best_params: dict[str, Any] | None = None
    n_trials: int | None = None
