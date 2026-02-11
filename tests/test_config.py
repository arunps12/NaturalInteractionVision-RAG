"""Tests for config (Pydantic settings + artifacts)."""

from visionllm_interactionanalysis.config import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelTrainerConfig,
    PipelineConfig,
)


def test_pipeline_config_creates_run_dir():
    cfg = PipelineConfig()
    assert cfg.timestamp
    assert cfg.resolved_run_dir.endswith(cfg.timestamp)


def test_pipeline_config_fixed_run_dir():
    cfg = PipelineConfig(run_dir="/tmp/fixed_run")
    assert cfg.resolved_run_dir == "/tmp/fixed_run"


def test_stage_configs_from_pipeline():
    pipe = PipelineConfig()
    ing = DataIngestionConfig.from_pipeline(pipe)
    val = DataValidationConfig.from_pipeline(pipe)
    tx = DataTransformationConfig.from_pipeline(pipe)
    mt = ModelTrainerConfig.from_pipeline(pipe)

    assert pipe.timestamp in ing.stage_dir
    assert pipe.timestamp in val.stage_dir
    assert pipe.timestamp in tx.stage_dir
    assert pipe.timestamp in mt.stage_dir

    assert ing.manifest_file.endswith("data_manifest.yaml")
    assert val.report_file.endswith("data_validation_report.yaml")
    assert tx.training_manifest_file.endswith("training_manifest.yaml")
    assert mt.best_model_file.endswith("fasterrcnn_best.pt")
