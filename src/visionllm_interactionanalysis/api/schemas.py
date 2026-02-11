"""Pydantic schemas for FastAPI request/response validation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    label: str
    category_id: int
    score: float
    bbox_xyxy: list[float]


class SpatialRelationResult(BaseModel):
    subject: str
    relation: str
    obj: str


class SceneResult(BaseModel):
    image_id: str
    objects: list[DetectionResult]
    spatial_relations: list[SpatialRelationResult]
    object_summary: list[str]


class ReasoningResult(BaseModel):
    interaction_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    involved_entities: list[str]
    evidence: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str


class PredictionResponse(BaseModel):
    """Full prediction response — detection + optional reasoning."""
    scene: SceneResult
    reasoning: ReasoningResult | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = False
    gpu_available: bool = False
