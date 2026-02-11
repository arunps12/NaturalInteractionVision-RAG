"""Tests for LLaVA reasoning output validation."""


from visionllm_interactionanalysis.reasoning.llava_engine import (
    ReasoningOutput,
    _extract_json,
)


def test_reasoning_output_valid():
    r = ReasoningOutput(
        interaction_type="caregiver_guided_attention",
        confidence=0.87,
        involved_entities=["adult", "child"],
        evidence={"action": "holding toy"},
        reasoning_summary="Adult guides child attention.",
    )
    assert r.confidence == 0.87


def test_reasoning_output_clamps_confidence():
    r = ReasoningOutput(
        interaction_type="test",
        confidence=1.5,
        involved_entities=[],
        reasoning_summary="test",
    )
    assert r.confidence == 1.0


def test_extract_json_plain():
    text = '{"interaction_type": "test", "confidence": 0.5}'
    result = _extract_json(text)
    assert result["interaction_type"] == "test"


def test_extract_json_fenced():
    text = "```json\n{\"interaction_type\": \"test\"}\n```"
    result = _extract_json(text)
    assert result["interaction_type"] == "test"


def test_extract_json_embedded():
    text = "Here is the output: {\"interaction_type\": \"foo\"} end."
    result = _extract_json(text)
    assert result["interaction_type"] == "foo"
