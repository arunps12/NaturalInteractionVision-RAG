"""CLI entry point for VisionLLM InteractionAnalysis.

Usage:
    python main.py train          # run training pipeline (stages 1-4)
    python main.py predict ...    # single-image prediction
    python main.py serve          # launch FastAPI server
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="visionllm",
        description="VisionLLM InteractionAnalysis CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # ── train ─────────────────────────────────────────────────────
    sub.add_parser("train", help="Run the full training pipeline (Stages 1-4)")

    # ── predict ───────────────────────────────────────────────────
    pred = sub.add_parser("predict", help="Run prediction on a single image")
    pred.add_argument("--image", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--num-classes", type=int, default=81)
    pred.add_argument("--score-threshold", type=float, default=0.5)
    pred.add_argument("--llava", action="store_true")
    pred.add_argument("--output", default=None)

    # ── serve ─────────────────────────────────────────────────────
    srv = sub.add_parser("serve", help="Launch FastAPI server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()

    if args.command == "train":
        from visionllm_interactionanalysis.pipeline.training_pipeline import run_training_pipeline
        run_training_pipeline()

    elif args.command == "predict":
        from visionllm_interactionanalysis.pipeline.prediction_pipeline import main as pred_main
        sys.argv = [
            "predict",
            "--image", args.image,
            "--checkpoint", args.checkpoint,
            "--num-classes", str(args.num_classes),
            "--score-threshold", str(args.score_threshold),
        ]
        if args.llava:
            sys.argv.append("--llava")
        if args.output:
            sys.argv.extend(["--output", args.output])
        pred_main()

    elif args.command == "serve":
        import uvicorn
        uvicorn.run(
            "visionllm_interactionanalysis.api.app:app",
            host=args.host,
            port=args.port,
            reload=False,
        )

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
