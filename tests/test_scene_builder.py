"""Tests for the symbolic scene builder."""

from visionllm_interactionanalysis.reasoning.scene_builder import (
    SymbolicScene,
    build_symbolic_scene,
)


def test_build_scene_filters_by_threshold():
    scene = build_symbolic_scene(
        image_id=42,
        boxes=[[10, 10, 100, 200], [200, 200, 300, 300]],
        labels=[1, 1],
        scores=[0.9, 0.3],
        score_threshold=0.5,
        category_names={1: "person"},
    )
    assert isinstance(scene, SymbolicScene)
    assert len(scene.objects) == 1
    assert scene.objects[0].label == "person"


def test_build_scene_spatial_relations():
    scene = build_symbolic_scene(
        image_id=1,
        boxes=[[10, 10, 50, 50], [200, 10, 250, 50]],
        labels=[1, 77],
        scores=[0.95, 0.85],
        score_threshold=0.5,
        category_names={1: "person", 77: "teddy bear"},
    )
    assert len(scene.objects) == 2
    rel_types = [r.relation for r in scene.spatial_relations]
    assert "left_of" in rel_types


def test_build_scene_empty_when_low_scores():
    scene = build_symbolic_scene(
        image_id=0,
        boxes=[[0, 0, 10, 10]],
        labels=[1],
        scores=[0.01],
        score_threshold=0.5,
    )
    assert len(scene.objects) == 0
