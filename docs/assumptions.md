# Assumptions

Documented assumptions made during implementation.

---

## Data

1. **COCO-2017 dataset** is available via `kagglehub` under the key `awsaf49/coco-2017-dataset`.
   If unavailable, set `VISIONLLM_DATASET_NAME` to an alternate Kaggle dataset handle.

2. **Data staging** writes to `$VISIONLLM_DATA_ROOT` (defaults to `$SCRATCH/visionllm_interaction/data`).
   On HPC nodes, `$SCRATCH` should be set; locally it falls back to `/tmp`.

3. **Both train and val splits are required** for the pipeline to proceed past validation.

## Model

4. **Faster R-CNN** uses 81 classes (COCO 80 categories + background).
   Category mapping is contiguous starting at 1.

5. **Validation loss** is used as the HPO objective metric (minimize).
   Faster R-CNN's `model.train()` mode is used during validation to compute losses
   (the model cannot compute detection losses in `eval()` mode).

6. **Optuna TPE sampler** is used for HPO. Only `batch_size` (categorical)
   and `lr` (log-uniform) are tuned.

## LLaVA Reasoning (Stage 5)

7. **Model:** `llava-hf/llava-v1.6-mistral-7b-hf` loaded with 4-bit quantization
   via `bitsandbytes` (`nf4` type, `float16` compute dtype).

8. **Hardware:** Requires ≥11 GB VRAM (RTX 2080 Ti minimum) for inference.
   If GPU memory is insufficient, the model will fail to load.

9. **JSON parsing:** The LLM output is parsed with up to 3 retry attempts.
   Regex extraction handles markdown-fenced and embedded JSON.

10. **Scene builder** spatial relations are heuristic (based on bbox center offsets).
    They are not ground-truth annotations.

## API

11. **FastAPI** serves on port 8000 by default. The model checkpoint path must be
    provided via `VISIONLLM_CHECKPOINT` environment variable.

12. **Batch endpoint** does not support LLaVA reasoning (detection only) to
    avoid GPU memory pressure from concurrent LLM inference.

## Deployment

13. **AWS deployment** assumes:
    - ECR repository named `visionllm-interactionanalysis` exists
    - ECS cluster `visionllm-cluster` and service `visionllm-service` exist
    - IAM role ARN is stored in `AWS_ROLE_ARN` GitHub secret
    - For GPU inference in production, use EC2-backed ECS tasks (not Fargate)

14. **CI/CD** triggers on push/PR to `main` branch. Deploy job only runs on push.

## Testing

15. **Unit tests** do not require GPU or real COCO data. They use synthetic
    fixtures with minimal COCO JSON structures.

16. **Integration tests** (training pipeline end-to-end) require the COCO
    dataset to be staged — these should be run manually or in a CI environment
    with data access.
