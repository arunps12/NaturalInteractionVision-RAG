"""Stage 3 — Data Transformation.

Reads the ingestion manifest and validation report, drops invalid
annotation IDs, writes cleaned COCO JSONs and a training manifest.
Raw data is never modified.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from visionllm_interactionanalysis.config.artifacts import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from visionllm_interactionanalysis.config.settings import DataTransformationConfig
from visionllm_interactionanalysis.utils import (
    PipelineException,
    ensure_dir,
    get_logger,
    load_json,
    read_yaml,
    write_json,
    write_yaml,
)

logger = get_logger(__name__)


class DataTransformation:
    """Clean annotations and produce a training manifest."""

    def __init__(
        self,
        config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
        validation_artifact: DataValidationArtifact,
    ) -> None:
        self.cfg = config
        self.ingestion = ingestion_artifact
        self.validation = validation_artifact

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _extract_invalid_ids(report: dict[str, Any], split: str) -> set[int]:
        stats = report.get(f"{split}_stats", {}) or {}
        ids = stats.get("invalid_bbox_ann_ids", []) or []
        result: set[int] = set()
        for x in ids:
            if isinstance(x, int):
                result.add(x)
            elif isinstance(x, str) and x.isdigit():
                result.add(int(x))
        if not result:
            # Fallback to examples
            for ex in stats.get("invalid_bbox_examples", []) or []:
                aid = ex.get("ann_id")
                if isinstance(aid, int):
                    result.add(aid)
        return result

    @staticmethod
    def _drop_by_ids(coco: dict[str, Any], drop: set[int]) -> tuple[dict[str, Any], int]:
        if not drop:
            return coco, 0
        before = len(coco.get("annotations", []))
        coco["annotations"] = [a for a in coco["annotations"] if a.get("id") not in drop]
        return coco, before - len(coco["annotations"])

    # ── main entry ───────────────────────────────────────────────────
    def run(self) -> DataTransformationArtifact:
        try:
            logger.info("===== Stage 3: Data Transformation Started =====")
            ensure_dir(self.cfg.stage_dir)
            ensure_dir(self.cfg.cleaned_annotation_dir)

            # 1) Read manifests
            manifest = read_yaml(self.ingestion.manifest_file)
            val_report = read_yaml(self.validation.report_file)

            if not val_report.get("validated", False):
                raise PipelineException("Validation report says dataset is NOT validated")

            # 2) Determine IDs to drop
            drop_train = self._extract_invalid_ids(val_report, "train")
            drop_val = self._extract_invalid_ids(val_report, "val")
            logger.info("Invalid IDs to drop — train: %d, val: %d", len(drop_train), len(drop_val))

            # 3) Load, clean, write
            train_coco = load_json(manifest["annotations"]["train"])
            val_coco = load_json(manifest["annotations"]["val"])

            train_coco, n_train = self._drop_by_ids(train_coco, drop_train)
            val_coco, n_val = self._drop_by_ids(val_coco, drop_val)
            logger.info("Dropped — train: %d, val: %d", n_train, n_val)

            write_json(self.cfg.cleaned_train_ann_file, train_coco)
            write_json(self.cfg.cleaned_val_ann_file, val_coco)

            # 4) Training manifest
            training_manifest = {
                "dataset_name": manifest.get("dataset_name"),
                "dataset_format": manifest.get("dataset_format"),
                "created_at": datetime.now().isoformat(),
                "source_manifest": self.ingestion.manifest_file,
                "transformation": {
                    "purpose": "clean_annotations_only",
                    "validation_report": self.validation.report_file,
                },
                "paths": {
                    "train_images": manifest["paths"]["train_images"],
                    "val_images": manifest["paths"]["val_images"],
                },
                "annotations": {
                    "train": self.cfg.cleaned_train_ann_file,
                    "val": self.cfg.cleaned_val_ann_file,
                },
            }
            write_yaml(self.cfg.training_manifest_file, training_manifest)
            logger.info("Training manifest → %s", self.cfg.training_manifest_file)
            logger.info("===== Stage 3: Data Transformation Complete =====")

            return DataTransformationArtifact(
                stage_dir=self.cfg.stage_dir,
                cleaned_train_ann_file=self.cfg.cleaned_train_ann_file,
                cleaned_val_ann_file=self.cfg.cleaned_val_ann_file,
                training_manifest_file=self.cfg.training_manifest_file,
                dropped_train_count=n_train,
                dropped_val_count=n_val,
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException("Data transformation failed", e)
