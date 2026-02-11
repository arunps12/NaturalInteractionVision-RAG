# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] — 2026-02-11

### Added
- **Stage 5:** Symbolic scene builder (`reasoning/scene_builder.py`) converts detections to structured scene graphs
- **Stage 5:** LLaVA-NeXT 7B reasoning engine (`reasoning/llava_engine.py`) with 4-bit quantization and retry logic
- **Stage 6:** FastAPI inference API with `/health`, `/predict`, and `/predict/batch` endpoints
- **Stage 7:** GPU-ready Dockerfile (CUDA 12.4, Python 3.11)
- **Stage 8:** GitHub Actions CI/CD pipeline (lint → test → ECR push → ECS deploy)
- Pydantic-validated config and artifact models
- CLI with `train`, `predict`, `serve` subcommands
- Unit tests for config, utils, scene builder, reasoning, and API
- Documentation: `assumptions.md`, `refactor_report.md`, `CHANGELOG.md`
- `.github/copilot-instructions.md` for AI agent guidance

### Changed
- **Project structure:** migrated from `visionllm_interaction/` to `src/visionllm_interactionanalysis/` (src-layout)
- **Config system:** dataclasses → Pydantic `BaseModel` with factory methods
- **Constants:** all hardcoded `/scratch` paths → environment-variable-configurable
- **Exception:** `CustomException` → `PipelineException`
- **Logger:** added console handler, configurable log directory
- **pyproject.toml:** added FastAPI, Pydantic, uvicorn, dev/llm optional deps
- **main.py:** monolithic script → CLI subcommands

### Removed
- `app.py` (empty), `demo.py` (empty), `template.py` (scaffolding script)
- `visionllm_interaction/` (old package layout)
- Empty placeholder files: `model_evaluation.py`, `model_pusher.py`
- `pipline/` directory (typo — replaced by `pipeline/`)
- `configuration/` directory (unused)

### Fixed
- Typo: `pipline` → `pipeline`
- Hardcoded paths to `/scratch/users/arunps`

## [0.1.0] — 2025-12-31

### Added
- Initial implementation of Stages 1–4 (data ingestion, validation, transformation, model training)
- Faster R-CNN with Optuna HPO and MLflow/DagsHub tracking
- COCO-2017 dataset support via kagglehub
