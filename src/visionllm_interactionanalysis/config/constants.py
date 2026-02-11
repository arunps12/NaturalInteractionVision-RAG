"""Pipeline constants — all paths configurable via env vars."""

from __future__ import annotations

import os

# ── Dataset ──────────────────────────────────────────────────────────
DATASET_NAME: str = os.environ.get("VISIONLLM_DATASET_NAME", "awsaf49/coco-2017-dataset")

# ── Global directories ───────────────────────────────────────────────
ARTIFACTS_DIR: str = os.environ.get("VISIONLLM_ARTIFACTS_DIR", "artifacts")
CONFIGS_DIR: str = os.environ.get("VISIONLLM_CONFIGS_DIR", "configs")
PIPELINE_NAME: str = "visionllm_interaction_pipeline"

# ── Data staging root (configurable — no hardcoded /scratch) ─────────
DATA_ROOT: str = os.environ.get(
    "VISIONLLM_DATA_ROOT",
    os.path.join(os.environ.get("SCRATCH", "/tmp"), "visionllm_interaction", "data"),
)
RAW_DATA_DIR: str = os.path.join(DATA_ROOT, "raw")

# ── COCO sub-directories (relative to RAW_DATA_DIR) ─────────────────
RAW_TRAIN_IMAGE_DIR: str = os.path.join(RAW_DATA_DIR, "train2017")
RAW_VAL_IMAGE_DIR: str = os.path.join(RAW_DATA_DIR, "val2017")
RAW_ANNOTATION_DIR: str = os.path.join(RAW_DATA_DIR, "annotations")
RAW_TRAIN_ANN_FILE: str = os.path.join(RAW_ANNOTATION_DIR, "instances_train2017.json")
RAW_VAL_ANN_FILE: str = os.path.join(RAW_ANNOTATION_DIR, "instances_val2017.json")

# ── Stage directory names ────────────────────────────────────────────
DATA_INGESTION_DIR: str = "data_ingestion"
DATA_VALIDATION_DIR: str = "data_validation"
DATA_TRANSFORMATION_DIR: str = "data_transformation"
MODEL_TRAINER_DIR: str = "model_trainer"

# ── Artifact file names ─────────────────────────────────────────────
DATA_INGESTION_MANIFEST: str = "data_manifest.yaml"
VALIDATION_REPORT: str = "data_validation_report.yaml"
CLEANED_TRAIN_ANN: str = "instances_train2017.cleaned.json"
CLEANED_VAL_ANN: str = "instances_val2017.cleaned.json"
TRAINING_MANIFEST: str = "training_manifest.yaml"
TRAINING_REPORT: str = "training_report.yaml"
BEST_MODEL_FILE: str = "fasterrcnn_best.pt"
LAST_MODEL_FILE: str = "fasterrcnn_last.pt"

# ── Config files ─────────────────────────────────────────────────────
MODEL_CONFIG_FILE: str = os.path.join(CONFIGS_DIR, "model.yaml")
SCHEMA_FILE: str = os.path.join(CONFIGS_DIR, "schema.yaml")

# ── Validation policy defaults ───────────────────────────────────────
DROP_INVALID_BBOX: bool = True
MAX_INVALID_BBOX_ALLOWED: int = 100
INGESTION_MODE: str = "register"
