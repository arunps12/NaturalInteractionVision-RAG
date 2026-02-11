# Copilot Instructions — VisionLLM InteractionAnalysis

## Project Overview

End-to-end vision + LLM pipeline for child–caregiver interaction analysis.
Faster R-CNN (ResNet-50 FPN) detects objects in COCO-2017 images, a symbolic
scene builder computes spatial relations, and LLaVA-NeXT 7B generates
natural-language reasoning about interactions.

## Repository Layout

```
src/visionllm_interactionanalysis/   ← importable package
  config/          settings.py (Pydantic configs), artifacts.py (Pydantic artifacts), constants.py
  detection/       data_ingestion.py, data_validation.py, data_transformation.py, model_trainer.py
  reasoning/       scene_builder.py, llava_engine.py
  pipeline/        training_pipeline.py, prediction_pipeline.py
  api/             app.py (FastAPI), schemas.py
  utils/           logger.py, exception.py, io_utils.py, ml_utils.py
tests/             pytest tests
configs/           model.yaml, schema.yaml (runtime YAML)
docs/              assumptions.md, refactor_report.md
```

## Key Conventions

| Area | Convention |
|------|-----------|
| Python version | ≥ 3.10 (target 3.11 for ruff) |
| Package manager | `uv` (not pip directly) |
| Config / validation | **Pydantic v2 `BaseModel`** — no raw dataclasses for configs or artifacts |
| Paths | Never hardcode `/scratch/…` — use env vars (`VISIONLLM_DATA_ROOT`, `SCRATCH`) or `PipelineConfig.from_env()` |
| Logging | `visionllm_interactionanalysis.utils.logger.get_logger(__name__)` |
| Exceptions | `PipelineException` from `utils.exception` — always wrap with context |
| YAML / JSON I/O | `read_yaml`, `write_yaml`, `load_json`, `write_json` from `utils.io_utils` |
| ML utilities | `set_seed`, `get_device`, `save_checkpoint`, `load_checkpoint` from `utils.ml_utils` |
| Linter | `ruff` (line-length 120, rules E/F/I/W, target py311) |
| Tests | `pytest` — `testpaths = ["tests"]`, `pythonpath = ["src"]` |
| Imports | Use absolute imports (`from visionllm_interactionanalysis.config import …`) |

## Pipeline Stages

1. **Data Ingestion** — `DataIngestion.run()` → `DataIngestionArtifact`
2. **Data Validation** — `DataValidation.run()` → `DataValidationArtifact`
3. **Data Transformation** — `DataTransformation.run()` → `DataTransformationArtifact`
4. **Model Training** — `ModelTrainer.run()` → `ModelTrainerArtifact` (Optuna HPO + MLflow)
5. **Symbolic Scene + LLaVA Reasoning** — `build_symbolic_scene()` + `LLaVAReasoningEngine`
6. **FastAPI** — `/health`, `/predict`, `/predict/batch`
7. **Docker** — GPU-ready (nvidia/cuda:12.4.1)
8. **AWS ECR/ECS + GitHub Actions** — CI/CD in `.github/workflows/ci-cd.yml`

## When Writing New Code

- Every stage class takes a Pydantic config and returns a Pydantic artifact.
- Run `ruff check src/ tests/` and `pytest tests/ -v` before committing.
- Use conventional commits: `feat(stageN): …`, `fix: …`, `docs: …`, `chore: …`.
- Document assumptions in `docs/assumptions.md`.
- Artifacts go under `artifacts/<timestamp>/<stage_name>/`.
- Model configs live in `configs/model.yaml`; dataset schema in `configs/schema.yaml`.
- For experiment tracking: MLflow + DagsHub (`dagshub.init` in model_trainer).
- For HPO: Optuna TPE sampler (batch_size + lr search space in model.yaml).

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `SCRATCH` | platform-dependent | Base scratch dir |
| `VISIONLLM_DATA_ROOT` | `$SCRATCH/data` | Raw dataset root |
| `VISIONLLM_LOGS_DIR` | `logs/` | Log file directory |
| `VISIONLLM_CHECKPOINT` | (none) | Checkpoint path for API inference |
| `DAGSHUB_REPO_OWNER` | `arunps12` | DagsHub repo owner |
| `DAGSHUB_REPO_NAME` | `VisionLLM_InteractionAnalysis` | DagsHub repo name |
| `AWS_REGION` | `eu-north-1` | AWS deployment region |

## Do NOT

- Use `dataclass` for pipeline configs/artifacts (use Pydantic).
- Import from the old `visionllm_interaction` package (deleted).
- Hardcode absolute paths like `/scratch/users/arunps/…`.
- Use `print()` for logging — use `logger.info/warning/error`.
- Skip type hints on public functions.
