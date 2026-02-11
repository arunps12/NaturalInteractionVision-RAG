"""Training pipeline — orchestrates Stages 1-4.

Usage:
    python -m visionllm_interactionanalysis.pipeline.training_pipeline
"""

from __future__ import annotations

from visionllm_interactionanalysis.config import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    PipelineConfig,
)
from visionllm_interactionanalysis.detection.data_ingestion import DataIngestion
from visionllm_interactionanalysis.detection.data_transformation import DataTransformation
from visionllm_interactionanalysis.detection.data_validation import DataValidation
from visionllm_interactionanalysis.detection.model_trainer import ModelTrainer
from visionllm_interactionanalysis.utils import get_logger

logger = get_logger(__name__)


def run_training_pipeline() -> None:
    """Execute the full training pipeline (Stages 1-4)."""

    logger.info("=== VisionLLM Training Pipeline Started ===")
    pipeline = PipelineConfig()
    logger.info("Run directory: %s", pipeline.resolved_run_dir)

    # Stage 1: Data Ingestion
    ing_cfg = DataIngestionConfig.from_pipeline(pipeline)
    ing_artifact = DataIngestion(ing_cfg).run()
    logger.info("Stage 1 complete — manifest: %s", ing_artifact.manifest_file)

    # Stage 2: Data Validation
    val_cfg = DataValidationConfig.from_pipeline(pipeline)
    val_artifact = DataValidation(val_cfg, ing_artifact).run()
    logger.info("Stage 2 complete — report: %s", val_artifact.report_file)

    # Stage 3: Data Transformation
    tx_cfg = DataTransformationConfig.from_pipeline(pipeline)
    tx_artifact = DataTransformation(tx_cfg, ing_artifact, val_artifact).run()
    logger.info("Stage 3 complete — training manifest: %s", tx_artifact.training_manifest_file)

    # Stage 4: Model Trainer
    mt_cfg = ModelTrainerConfig.from_pipeline(pipeline)
    mt_artifact = ModelTrainer(mt_cfg, tx_artifact).run()
    logger.info(
        "Stage 4 complete — best model: %s (val_loss=%.6f)",
        mt_artifact.best_model_path, mt_artifact.best_metric_value,
    )

    logger.info("=== VisionLLM Training Pipeline Finished ===")


if __name__ == "__main__":
    run_training_pipeline()
