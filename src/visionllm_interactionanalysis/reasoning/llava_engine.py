"""Stage 5b — LLaVA-NeXT 7B multimodal reasoning engine.

Loads llava-hf/llava-v1.6-mistral-7b-hf in 4-bit quantization,
takes a symbolic scene + image, and produces structured JSON reasoning.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from visionllm_interactionanalysis.reasoning.scene_builder import SymbolicScene
from visionllm_interactionanalysis.utils import PipelineException, get_logger

logger = get_logger(__name__)

# ── Structured output schema ─────────────────────────────────────────

class ReasoningOutput(BaseModel):
    """Validated structured output from the LLM reasoning engine."""

    interaction_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    involved_entities: list[str]
    evidence: dict[str, Any] = Field(default_factory=dict)
    reasoning_summary: str

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


# ── Prompt template ──────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert scene understanding and interaction analysis agent.
Given a symbolic scene description from an object detection model and
the original image, produce a structured JSON analysis of the
interactions depicted.

Output ONLY valid JSON matching this schema:
{
  "interaction_type": "<string>",
  "confidence": <float 0-1>,
  "involved_entities": ["<entity1>", ...],
  "evidence": {"<key>": "<value>", ...},
  "reasoning_summary": "<one paragraph>"
}

Focus on child–caregiver interactions when detected. If no humans are
present, describe object-level interactions instead.
"""


def _build_user_prompt(scene: SymbolicScene) -> str:
    objs = ", ".join(f"{o.label} (score={o.score})" for o in scene.objects)
    rels = "; ".join(f"{r.subject} {r.relation} {r.obj}" for r in scene.spatial_relations)
    return (
        f"Image ID: {scene.image_id}\n"
        f"Detected objects: {objs}\n"
        f"Spatial relations: {rels or 'none inferred'}\n\n"
        "Analyze the scene and return the structured JSON output."
    )


# ── JSON parsing with retry ─────────────────────────────────────────

def _extract_json(text: str) -> dict[str, Any]:
    """Try to extract JSON from LLM output, handling markdown fences."""
    # Try direct parse first
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ```
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise PipelineException(f"Cannot parse JSON from LLM output: {text[:200]}...")


# ── LLaVA engine ─────────────────────────────────────────────────────

class LLaVAReasoningEngine:
    """LLaVA-NeXT 7B 4-bit reasoning engine for scene interpretation."""

    MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"

    def __init__(self, model_id: str | None = None, max_retries: int = 3):
        self.model_id = model_id or self.MODEL_ID
        self.max_retries = max_retries
        self._model = None
        self._processor = None

    def load(self) -> None:
        """Lazy-load the model with 4-bit quantization."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import BitsAndBytesConfig, LlavaNextForConditionalGeneration, LlavaNextProcessor

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            logger.info("Loading LLaVA model: %s (4-bit)", self.model_id)
            self._processor = LlavaNextProcessor.from_pretrained(self.model_id)
            self._model = LlavaNextForConditionalGeneration.from_pretrained(
                self.model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            logger.info("LLaVA model loaded successfully")
        except Exception as e:
            raise PipelineException(f"Failed to load LLaVA model: {self.model_id}", e)

    def reason(
        self,
        scene: SymbolicScene,
        image: Any | None = None,
    ) -> ReasoningOutput:
        """Generate structured reasoning for a symbolic scene.

        Args:
            scene: SymbolicScene from the scene builder
            image: PIL Image (optional, for multimodal input)
        """
        self.load()
        import torch

        user_prompt = _build_user_prompt(scene)
        conversation = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if image is not None:
                    prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self._processor(images=image, text=prompt, return_tensors="pt").to(self._model.device)
                else:
                    prompt = self._processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)

                with torch.no_grad():
                    output_ids = self._model.generate(**inputs, max_new_tokens=512, do_sample=False)

                generated = self._processor.decode(output_ids[0], skip_special_tokens=True)
                # Extract only the assistant's response
                if "[/INST]" in generated:
                    generated = generated.split("[/INST]")[-1].strip()

                parsed = _extract_json(generated)
                result = ReasoningOutput(**parsed)
                logger.info("Reasoning succeeded on attempt %d for image %s", attempt, scene.image_id)
                return result

            except Exception as e:
                last_error = e
                logger.warning("Reasoning attempt %d/%d failed: %s", attempt, self.max_retries, e)

        raise PipelineException(
            f"LLaVA reasoning failed after {self.max_retries} attempts for image {scene.image_id}",
            last_error,
        )
