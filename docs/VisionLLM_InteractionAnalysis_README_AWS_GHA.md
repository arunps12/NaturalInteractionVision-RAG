# VisionLLM_InteractionAnalysis

![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![COCO](https://img.shields.io/badge/Dataset-COCO--2017-blue)
![MLflow](https://img.shields.io/badge/Experiment-MLflow-green)
![DagsHub](https://img.shields.io/badge/Tracking-DagsHub-orange)
![License](https://img.shields.io/badge/License-MIT--2.0-lightgrey)

------------------------------------------------------------------------

## Overview

VisionLLM_InteractionAnalysis is a modular, artifact-driven computer
vision and multimodal reasoning framework designed for structured
analysis of naturalistic child--caregiver interaction imagery and video.

The system integrates:

-   Faster R-CNN object detection (COCO-pretrained and fine-tuned)
-   Structured symbolic scene representation
-   LLaVA-NeXT (7B, 4-bit) multimodal reasoning engine
-   Manifest-driven reproducible pipeline
-   MLflow experiment tracking (DagsHub compatible)
-   cloud-ready and containerizable architecture

This repository follows research-grade reproducibility standards and is
designed to scale from experimentation to deployment.

------------------------------------------------------------------------

## Full Pipeline Architecture

``` mermaid
flowchart TD
    A[Raw COCO Dataset] --> B[Data Ingestion]
    B -->|data_manifest.yaml| C[Data Validation]
    C -->|validation_report.yaml| D[Data Transformation]
    D -->|training_manifest.yaml| E[Model Trainer]
    E -->|trained checkpoints| F[Symbolic Scene Builder]
    F --> G[LLM Reasoning Engine (LLaVA-NeXT 7B)]
    G --> H[Structured JSON Outputs]
    H --> I[API Layer]
    I --> J[Docker & Deployment]
```

------------------------------------------------------------------------

# Pipeline Stages

## Stage 1: Data Ingestion

Goal: tage raw COCO-style datasets and create canonical dataset manifests in a reproducible and idempotent manner.

-   Checks whether COCO train/val splits already exist in the specified storage path
-   If datasets are already present, skips download and performs integrity verification only
-   If missing, downloads and validates COCO train/val splits using kagglehub
-   Verifies directory structure and annotation availability
-   Stages images and annotations to configured storage (uio ml1 node scratch folder path)
-   Generates a deterministic dataset manifest (data_manifest.yaml)
-   Ensures idempotent execution (safe to re-run without duplicating data)
-   Guarantees reproducible data loading across environments

Artifacts: -
artifacts/`<timestamp>`{=html}/data_ingestion/data_manifest.yaml

------------------------------------------------------------------------

## Stage 2: Data Validation

Goal: Guarantee dataset integrity before model training.

-   Validates manifest schema
-   Ensures annotation-image consistency
-   Detects invalid bounding boxes
-   Schema validation against COCO format
-   Produces structured validation report

Artifacts: -
artifacts/`<timestamp>`{=html}/data_validation/data_validation_report.yaml

------------------------------------------------------------------------

## Stage 3: Data Transformation

Goal: Prepare a clean training-ready dataset while preserving raw
immutability.

-   Drops invalid annotations only (no raw overwrite)
-   Generates cleaned COCO JSON files
-   Writes training manifest
-   Maintains full traceability

Artifacts: - instances_train2017.cleaned.json -
instances_val2017.cleaned.json - training_manifest.yaml

------------------------------------------------------------------------

## Stage 4: Model Trainer

Goal: Train object detection model in reproducible and
experiment-tracked manner.

Model: - Faster R-CNN (ResNet-50 FPN backbone)

Features: - Config-driven training (model.yaml) - Optuna hyperparameter
optimization - MLflow tracking via DagsHub - Best and last checkpoint
saving - YAML-based training report generation

Artifacts: - fasterrcnn_best.pt - fasterrcnn_last.pt -
training_report.yaml

------------------------------------------------------------------------

## Stage 5: LLM-Based Scene Reasoning (Planned / In Progress)

Goal: Convert object detections into structured symbolic scene reasoning
outputs.

Components:

1.  Symbolic Scene Builder
    -   Converts bounding boxes to structured JSON
    -   Encodes spatial relationships (left_of, holding, looking_at)
    -   Creates object-level semantic graph
2.  Multimodal Reasoning Engine
    -   LLaVA-NeXT 7B (4-bit quantized for 11GB GPUs)
    -   Prompt-driven scene interpretation
    -   Structured JSON reasoning output
    -   Multilingual support (Hindi / Norwegian compatible)
3.  Output Format
    -   Scene summary
    -   Interaction classification
    -   Confidence scores
    -   Explainable reasoning trace

Example Structured Output:

{ "interaction_type": "caregiver_guided_attention", "confidence": 0.87,
"objects": \["adult", "child", "toy"\], "spatial_relations": \["adult
holding toy", "child looking_at toy"\], "reasoning_summary": "Adult
appears to guide child's attention toward toy." }

------------------------------------------------------------------------

## Stage 6: API Layer (Planned)

-   FastAPI service
-   Batch inference endpoints
-   JSON schema validation
-   Model version control
-   GPU-aware inference routing

------------------------------------------------------------------------

## Stage 7: Containerization

-   Dockerized training and inference
-   CUDA-compatible builds
-   Reproducible environments (uv / pip)
-   MLflow + DagsHub integration

------------------------------------------------------------------------

## Stage 8: Deployment (AWS + GitHub Actions)

-   **GitHub Actions CI/CD** for automated linting, tests, and builds on
    each PR/push
-   **Docker image build + push to Amazon ECR**
-   Deploy options (choose based on cost/latency needs):
    -   **ECS (Fargate or EC2)** for scalable API inference services
    -   **EC2 GPU instances** for heavier batch inference jobs
    -   **SageMaker** for managed endpoints and model registry workflows
-   **Monitoring** via CloudWatch + structured logs
-   **Artifact/versioning**: model checkpoints and manifests stored in
    S3 (and optionally tracked in MLflow)

------------------------------------------------------------------------

# Design Principles

-   Manifest-driven architecture
-   Immutable raw data
-   Strict validation gates
-   Explicit artifacts at every stage
-   Explainable AI outputs
-   Research-to-production scalability
-   HPC-compatible GPU workflows

------------------------------------------------------------------------

# Target Use Cases

-   Child--caregiver interaction research
-   Egocentric image and video analysis
-   Multimodal AI experimentation
-   Vision-language benchmarking
-   Transfer learning to domain-specific datasets

------------------------------------------------------------------------

# Hardware Compatibility

Local development / lab workstation: - RTX 2080 Ti (11GB) using 4-bit
LLaVA-NeXT 7B - CUDA 12.x environments

Cloud (AWS) options: - GPU inference/training using **NVIDIA T4 / A10G /
A100** class instances (depending on budget and throughput) - Container
images published to **Amazon ECR** and deployed via **GitHub Actions**

------------------------------------------------------------------------

# Author

Arun Prakash Singh\
Researcher, University of Oslo\
Computational Modeling of Child Language & Interaction AI

------------------------------------------------------------------------

License: MIT 2.0
