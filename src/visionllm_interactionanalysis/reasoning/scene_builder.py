"""Stage 5a — Symbolic Scene Builder.

Converts Faster R-CNN detection outputs (bounding boxes, labels, scores)
into a structured symbolic scene representation suitable for LLM reasoning.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from visionllm_interactionanalysis.utils import get_logger

logger = get_logger(__name__)

# ── COCO category id→name (subset relevant to interaction analysis) ──
COCO_CATEGORY_NAMES: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 15: "bench", 16: "bird",
    17: "cat", 18: "dog", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 35: "skis", 37: "sports ball",
    39: "baseball bat", 41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife",
    50: "spoon", 51: "bowl", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 67: "cell phone", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class DetectedObject(BaseModel):
    """Single detected object."""
    label: str
    category_id: int
    score: float
    bbox_xyxy: list[float] = Field(description="[x1, y1, x2, y2]")


class SpatialRelation(BaseModel):
    """Spatial relation between two objects."""
    subject: str
    relation: str
    obj: str


class SymbolicScene(BaseModel):
    """Structured scene representation."""
    image_id: int | str
    objects: list[DetectedObject]
    spatial_relations: list[SpatialRelation]
    object_summary: list[str]


def _compute_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _compute_spatial_relations(objects: list[DetectedObject]) -> list[SpatialRelation]:
    """Infer simple spatial relations from bounding boxes."""
    relations: list[SpatialRelation] = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i >= j:
                continue
            ca = _compute_center(a.bbox_xyxy)
            cb = _compute_center(b.bbox_xyxy)
            if ca[0] < cb[0] - 20:
                relations.append(SpatialRelation(subject=a.label, relation="left_of", obj=b.label))
            elif ca[0] > cb[0] + 20:
                relations.append(SpatialRelation(subject=a.label, relation="right_of", obj=b.label))
            if ca[1] < cb[1] - 20:
                relations.append(SpatialRelation(subject=a.label, relation="above", obj=b.label))

            # Overlap heuristic → "near"
            ax1, ay1, ax2, ay2 = a.bbox_xyxy
            bx1, by1, bx2, by2 = b.bbox_xyxy
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            if ix > 0 and iy > 0:
                relations.append(SpatialRelation(subject=a.label, relation="overlapping", obj=b.label))
    return relations


def build_symbolic_scene(
    image_id: int | str,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
    score_threshold: float = 0.5,
    category_names: dict[int, str] | None = None,
) -> SymbolicScene:
    """Build a SymbolicScene from raw detection outputs.

    Args:
        image_id: identifier for the image
        boxes: list of [x1,y1,x2,y2] bounding boxes
        labels: list of category IDs (contiguous or COCO)
        scores: list of confidence scores
        score_threshold: minimum score to include
        category_names: optional mapping from category_id → name
    """
    cat_names = category_names or COCO_CATEGORY_NAMES

    objects: list[DetectedObject] = []
    for box, lbl, sc in zip(boxes, labels, scores):
        if sc < score_threshold:
            continue
        name = cat_names.get(lbl, f"class_{lbl}")
        objects.append(DetectedObject(label=name, category_id=lbl, score=round(sc, 4), bbox_xyxy=box))

    relations = _compute_spatial_relations(objects)
    summary = list({o.label for o in objects})

    scene = SymbolicScene(
        image_id=image_id,
        objects=objects,
        spatial_relations=relations,
        object_summary=sorted(summary),
    )
    logger.debug("Built scene for image %s: %d objects, %d relations", image_id, len(objects), len(relations))
    return scene
