# Refactor Report

**Date:** 2026-02-11
**Author:** AI Agent (automated refactor)

---

## Summary

Migrated project from a flat `visionllm_interaction/` package layout to a
clean `src/visionllm_interactionanalysis/` src-layout, eliminating dead code,
hardcoded paths, and duplicate logic.

---

## What Was Deleted

| File/Directory | Reason |
|---|---|
| `app.py` | Empty placeholder (0 bytes) — replaced by `src/.../api/app.py` |
| `demo.py` | Empty placeholder (0 bytes) — no useful code |
| `template.py` | Scaffolding script used once to create initial files — no runtime value |
| `Dockerfile` (old) | Empty file — replaced with GPU-ready Dockerfile |
| `visionllm_interaction/` (entire dir) | Old package superseded by `src/visionllm_interactionanalysis/` |
| `visionllm_interaction/components/model_evaluation.py` | Empty placeholder (0 bytes) |
| `visionllm_interaction/components/model_pusher.py` | Empty placeholder (0 bytes) |
| `visionllm_interaction/pipline/` (note: typo "pipline") | Empty files; replaced by `src/.../pipeline/` |
| `visionllm_interaction/configuration/` | Empty `__init__.py` only — unused |
| `visionllm_interaction/constants/__init__.py` | Empty — constants moved to `config/constants.py` |
| `__pycache__/` dirs | Byte-compiled cache — should never be committed |

## What Was Moved / Renamed

| Old Location | New Location | Notes |
|---|---|---|
| `visionllm_interaction/components/data_ingestion.py` | `src/.../detection/data_ingestion.py` | Refactored to use Pydantic configs |
| `visionllm_interaction/components/data_validation.py` | `src/.../detection/data_validation.py` | Refactored |
| `visionllm_interaction/components/data_transformation.py` | `src/.../detection/data_transformation.py` | Refactored |
| `visionllm_interaction/components/model_trainer.py` | `src/.../detection/model_trainer.py` | Refactored |
| `visionllm_interaction/entity/config_entity.py` | `src/.../config/settings.py` | Now Pydantic `BaseModel`s |
| `visionllm_interaction/entity/artifact_entity.py` | `src/.../config/artifacts.py` | `dataclass` → Pydantic `BaseModel` |
| `visionllm_interaction/constants/training_pipeline/__init__.py` | `src/.../config/constants.py` | Env-var-configurable, no hardcoded `/scratch` |
| `visionllm_interaction/utils/main_utils.py` | `src/.../utils/io_utils.py` | Cleaner naming |
| `visionllm_interaction/utils/ml_utils.py` | `src/.../utils/ml_utils.py` | Minor cleanup |
| `visionllm_interaction/logger/logger.py` | `src/.../utils/logger.py` | Now in utils subpackage |
| `visionllm_interaction/exception/custom_exception.py` | `src/.../utils/exception.py` | Renamed `CustomException` → `PipelineException` |
| `config/` | `configs/` | Matches spec; old `config/` retained for backward compat |
| `main.py` | `main.py` | Rewritten with CLI subcommands (train/predict/serve) |
| `visionllm_interaction/pipline/` | `src/.../pipeline/` | Fixed typo "pipline" → "pipeline" |

## What Was Refactored

1. **Config entities** — dataclasses replaced with Pydantic `BaseModel`s;
   factory methods (`from_pipeline()`) replace manual `__init__` wiring.
2. **Constants** — all hardcoded `/scratch/users/arunps` paths replaced with
   `os.environ.get()` calls; configurable via `VISIONLLM_DATA_ROOT`, `SCRATCH`.
3. **Exception class** — renamed `CustomException` → `PipelineException`
   for clarity; format string simplified.
4. **Logger** — added console handler alongside file handler; log dir
   configurable via `VISIONLLM_LOGS_DIR`.
5. **Training pipeline** — extracted into `pipeline/training_pipeline.py`
   with a single `run_training_pipeline()` function (was inline in `main.py`).
6. **`main.py`** — rewritten as CLI with `train`, `predict`, `serve` subcommands.

## What Was Added (New)

| Path | Purpose |
|---|---|
| `src/visionllm_interactionanalysis/reasoning/scene_builder.py` | Stage 5a: symbolic scene builder |
| `src/visionllm_interactionanalysis/reasoning/llava_engine.py` | Stage 5b: LLaVA-NeXT 7B reasoning |
| `src/visionllm_interactionanalysis/api/app.py` | Stage 6: FastAPI inference service |
| `src/visionllm_interactionanalysis/api/schemas.py` | Pydantic request/response models |
| `src/visionllm_interactionanalysis/pipeline/prediction_pipeline.py` | Single-image prediction CLI |
| `Dockerfile` | Stage 7: GPU-ready container |
| `.github/workflows/ci-cd.yml` | Stage 8: CI/CD pipeline |
| `.github/task-definition.json` | ECS Fargate task definition |
| `tests/` | Unit tests for config, utils, scene builder, reasoning, API |
| `docs/assumptions.md` | Documented assumptions |
| `docs/refactor_report.md` | This file |

## Why

- **src-layout** is the Python packaging best practice; prevents import shadowing
- **Pydantic** provides schema validation, serialization, and IDE support
- **No hardcoded paths** ensures portability across HPC nodes, local dev, and cloud
- **Typo fix** (`pipline` → `pipeline`) prevents confusion
- **Dead code removal** reduces maintenance burden and cognitive load
- **CLI subcommands** replace the monolithic `main()` function
