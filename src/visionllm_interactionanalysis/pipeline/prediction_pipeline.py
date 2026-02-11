"""Prediction pipeline — load model + run inference + LLaVA reasoning.

Usage:
    python -m visionllm_interactionanalysis.pipeline.prediction_pipeline --image path/to/img.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from visionllm_interactionanalysis.reasoning.llava_engine import LLaVAReasoningEngine
from visionllm_interactionanalysis.reasoning.scene_builder import build_symbolic_scene
from visionllm_interactionanalysis.utils import PipelineException, get_logger
from visionllm_interactionanalysis.utils.ml_utils import load_checkpoint

logger = get_logger(__name__)


def _load_model(checkpoint: str, num_classes: int = 81) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    load_checkpoint(model, checkpoint)
    model.eval()
    return model


@torch.no_grad()
def predict_single(
    image_path: str,
    checkpoint: str,
    num_classes: int = 81,
    score_threshold: float = 0.5,
    use_llava: bool = False,
) -> dict:
    """Run detection + optional LLaVA reasoning on a single image."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint, num_classes).to(device)

    bgr = cv2.imread(image_path)
    if bgr is None:
        raise PipelineException(f"Cannot read image: {image_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    outputs = model(img_tensor)[0]
    boxes = outputs["boxes"].cpu().tolist()
    labels = outputs["labels"].cpu().tolist()
    scores = outputs["scores"].cpu().tolist()

    scene = build_symbolic_scene(
        image_id=Path(image_path).stem,
        boxes=boxes,
        labels=labels,
        scores=scores,
        score_threshold=score_threshold,
    )

    result = scene.model_dump()

    if use_llava:
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_path).convert("RGB")
        engine = LLaVAReasoningEngine()
        reasoning = engine.reason(scene, image=pil_img)
        result["reasoning"] = reasoning.model_dump()

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="VisionLLM Prediction Pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--num-classes", type=int, default=81)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--llava", action="store_true", help="Enable LLaVA reasoning")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    result = predict_single(
        image_path=args.image,
        checkpoint=args.checkpoint,
        num_classes=args.num_classes,
        score_threshold=args.score_threshold,
        use_llava=args.llava,
    )

    out_json = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(out_json, encoding="utf-8")
        logger.info("Result saved to %s", args.output)
    else:
        print(out_json)


if __name__ == "__main__":
    main()
