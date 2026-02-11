"""Stage 2 — COCO dataset validation.

Validates manifest schema, image/annotation consistency, bounding-box
integrity, and cross-split category alignment.  Produces a structured
YAML validation report.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from visionllm_interactionanalysis.config.artifacts import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from visionllm_interactionanalysis.config.settings import DataValidationConfig
from visionllm_interactionanalysis.utils import (
    PipelineException,
    ensure_dir,
    get_logger,
    load_json,
    read_yaml,
    write_yaml,
)

logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png"}


class DataValidation:
    """Validate COCO train+val dataset and write a report."""

    def __init__(
        self,
        config: DataValidationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        self.cfg = config
        self.ingestion = ingestion_artifact

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _list_images(img_dir: str) -> list[str]:
        if not os.path.isdir(img_dir):
            return []
        return [str(p) for p in Path(img_dir).rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]

    @staticmethod
    def _require_keys(obj: dict, keys: list[str], ctx: str) -> None:
        missing = [k for k in keys if k not in obj]
        if missing:
            raise PipelineException(f"Missing keys {missing} in {ctx}")

    def _validate_manifest(self) -> dict[str, Any]:
        path = self.ingestion.manifest_file
        if not os.path.exists(path):
            raise PipelineException(f"Manifest not found: {path}")
        m = read_yaml(path)
        self._require_keys(m, ["dataset_name", "dataset_format", "paths", "annotations"], "manifest")
        self._require_keys(m["paths"], ["train_images", "val_images"], "manifest.paths")
        self._require_keys(m["annotations"], ["train", "val"], "manifest.annotations")
        return m

    def _validate_coco_schema(self, coco: dict, split: str) -> None:
        for k in ("images", "annotations", "categories"):
            if k not in coco:
                raise PipelineException(f"COCO JSON missing '{k}' in {split}")
        if coco["categories"]:
            self._require_keys(coco["categories"][0], ["id", "name"], f"{split}.categories[0]")
        if coco["images"]:
            self._require_keys(coco["images"][0], ["id", "file_name", "width", "height"], f"{split}.images[0]")
        if coco["annotations"]:
            self._require_keys(
                coco["annotations"][0],
                ["id", "image_id", "category_id", "bbox", "area", "iscrowd"],
                f"{split}.annotations[0]",
            )

    def _validate_refs_and_bbox(
        self,
        coco: dict[str, Any],
        img_dir: str,
        split: str,
    ) -> dict[str, Any]:
        """Cross-reference + bbox validation; returns stats dict."""
        images = coco.get("images", [])
        annotations = coco.get("annotations", [])
        categories = coco.get("categories", [])

        img_by_id = {im["id"]: im for im in images}
        cat_ids = {c["id"] for c in categories}

        missing_files = bad_img_ref = bad_cat_ref = invalid_bbox = 0
        invalid_ids: set[int] = set()
        examples: list[dict] = []

        for im in images:
            fp = os.path.join(img_dir, im["file_name"])
            if not os.path.exists(fp):
                missing_files += 1

        for ann in annotations:
            iid, cid = ann.get("image_id"), ann.get("category_id")
            if iid not in img_by_id:
                bad_img_ref += 1
                continue
            if cid not in cat_ids:
                bad_cat_ref += 1
                continue
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                invalid_bbox += 1
                invalid_ids.add(ann["id"])
                if len(examples) < 10:
                    examples.append({"ann_id": ann["id"], "bbox": bbox, "reason": "malformed"})
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                invalid_bbox += 1
                invalid_ids.add(ann["id"])
                if len(examples) < 10:
                    examples.append({"ann_id": ann["id"], "bbox": bbox, "reason": "non-positive"})
                continue
            im_info = img_by_id[iid]
            if (x + w) > im_info["width"] or (y + h) > im_info["height"]:
                invalid_bbox += 1
                invalid_ids.add(ann["id"])
                if len(examples) < 10:
                    examples.append({"ann_id": ann["id"], "bbox": bbox, "reason": "out-of-bounds"})

        # Hard failures
        if missing_files:
            raise PipelineException(f"{split}: {missing_files} images missing on disk")
        if bad_img_ref:
            raise PipelineException(f"{split}: {bad_img_ref} annotations reference missing image_id")
        if bad_cat_ref:
            raise PipelineException(f"{split}: {bad_cat_ref} annotations reference missing category_id")

        # Soft bbox policy
        if invalid_bbox > 0:
            if self.cfg.drop_invalid_bbox and invalid_bbox <= self.cfg.max_invalid_bbox_allowed:
                before = len(annotations)
                coco["annotations"] = [a for a in annotations if a["id"] not in invalid_ids]
                logger.warning(
                    "%s: dropped %d invalid bbox annotations (limit=%d)",
                    split, before - len(coco["annotations"]), self.cfg.max_invalid_bbox_allowed,
                )
            else:
                raise PipelineException(
                    f"{split}: {invalid_bbox} invalid bboxes (limit={self.cfg.max_invalid_bbox_allowed})"
                )

        per_cat = Counter(a["category_id"] for a in coco["annotations"])
        imgs_with_ann = len({a["image_id"] for a in coco["annotations"]})

        return {
            "num_images": len(images),
            "num_annotations": len(coco["annotations"]),
            "num_categories": len(categories),
            "images_with_annotations": imgs_with_ann,
            "per_category_counts": dict(per_cat),
            "invalid_bbox_found": invalid_bbox,
            "invalid_bbox_examples": examples,
            "invalid_bbox_ann_ids": sorted(invalid_ids),
        }

    # ── main entry ───────────────────────────────────────────────────
    def run(self) -> DataValidationArtifact:
        report: dict[str, Any] = {
            "validation_status": "FAIL",
            "validated": False,
            "manifest_path": self.ingestion.manifest_file,
            "schema_path": self.cfg.schema_file,
            "policy": {
                "require_val": self.cfg.require_val,
                "drop_invalid_bbox": self.cfg.drop_invalid_bbox,
                "max_invalid_bbox_allowed": self.cfg.max_invalid_bbox_allowed,
            },
            "error": None,
        }
        try:
            logger.info("===== Stage 2: Data Validation Started =====")
            ensure_dir(self.cfg.stage_dir)

            if not os.path.exists(self.cfg.schema_file):
                raise PipelineException(f"Schema file missing: {self.cfg.schema_file}")

            manifest = self._validate_manifest()
            train_img_dir = manifest["paths"]["train_images"]
            val_img_dir = manifest["paths"]["val_images"]
            train_ann = manifest["annotations"]["train"]
            val_ann = manifest["annotations"]["val"]

            # Existence checks
            if not os.path.isdir(train_img_dir):
                raise PipelineException(f"Train images dir missing: {train_img_dir}")
            if not os.path.isfile(train_ann):
                raise PipelineException(f"Train annotation file missing: {train_ann}")
            if self.cfg.require_val:
                if not os.path.isdir(val_img_dir):
                    raise PipelineException(f"Val images dir missing: {val_img_dir}")
                if not os.path.isfile(val_ann):
                    raise PipelineException(f"Val annotation file missing: {val_ann}")

            train_imgs = self._list_images(train_img_dir)
            val_imgs = self._list_images(val_img_dir)
            if not train_imgs:
                raise PipelineException(f"No train images on disk: {train_img_dir}")
            if self.cfg.require_val and not val_imgs:
                raise PipelineException(f"No val images on disk: {val_img_dir}")

            train_coco = load_json(train_ann)
            val_coco = load_json(val_ann)

            self._validate_coco_schema(train_coco, "train")
            self._validate_coco_schema(val_coco, "val")

            # Category consistency
            train_cats = {(c["id"], c["name"]) for c in train_coco.get("categories", [])}
            val_cats = {(c["id"], c["name"]) for c in val_coco.get("categories", [])}
            if train_cats != val_cats:
                raise PipelineException("Train/val category mismatch")

            train_stats = self._validate_refs_and_bbox(train_coco, train_img_dir, "train")
            val_stats = self._validate_refs_and_bbox(val_coco, val_img_dir, "val")

            report.update({
                "validation_status": "PASS",
                "validated": True,
                "disk_counts": {"train_images": len(train_imgs), "val_images": len(val_imgs)},
                "train_stats": train_stats,
                "val_stats": val_stats,
            })
            logger.info("===== Stage 2: Data Validation Complete =====")

            return DataValidationArtifact(
                stage_dir=self.cfg.stage_dir,
                report_file=self.cfg.report_file,
                validated=True,
            )
        except PipelineException as pe:
            report["error"] = str(pe)
            logger.error("Validation failed: %s", pe)
            raise
        except Exception as e:
            report["error"] = str(e)
            raise PipelineException("Validation failed", e)
        finally:
            try:
                ensure_dir(self.cfg.stage_dir)
                write_yaml(self.cfg.report_file, report)
                logger.info("Report → %s", self.cfg.report_file)
            except Exception:
                logger.exception("Failed to write validation report")
