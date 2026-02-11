"""Stage 1 — Idempotent COCO data ingestion.

Downloads the COCO-2017 dataset via kagglehub (cached),
stages it to a configurable data root, and writes a manifest YAML.
Re-running is safe: existing files are never re-copied.
"""

from __future__ import annotations

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import kagglehub

from visionllm_interactionanalysis.config.artifacts import DataIngestionArtifact
from visionllm_interactionanalysis.config.settings import DataIngestionConfig
from visionllm_interactionanalysis.utils import PipelineException, ensure_dir, get_logger, write_yaml

logger = get_logger(__name__)


class DataIngestion:
    """Download / register COCO data and write an ingestion manifest."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.cfg = config

    # ── helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _is_zip(path: str) -> bool:
        return Path(path).suffix.lower() == ".zip" and Path(path).is_file()

    def _extract_zip(self, zip_path: str, dest: str) -> str:
        ensure_dir(dest)
        logger.info("Extracting %s → %s", zip_path, dest)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        return dest

    def _find_coco_root(self, base: str) -> str:
        """Walk ≤2 levels to find a dir with train2017/ and annotations/."""
        train_name = Path(self.cfg.raw_train_image_dir).name
        b = Path(base)
        candidates = [b] + list(b.glob("*")) + list(b.glob("*/*"))
        for c in candidates:
            if c.is_dir() and (c / train_name).exists() and (c / "annotations").exists():
                return str(c)
        raise PipelineException(
            f"Cannot locate COCO root under {base} "
            f"(expected '{train_name}/' + 'annotations/')"
        )

    @staticmethod
    def _copytree_if_missing(src: str, dst: str) -> None:
        if os.path.exists(dst):
            logger.info("Already staged (skip): %s", dst)
            return
        logger.info("Copying tree: %s → %s", src, dst)
        shutil.copytree(src, dst)

    # ── main entry ───────────────────────────────────────────────────
    def run(self) -> DataIngestionArtifact:
        """Execute data ingestion — idempotent."""
        try:
            logger.info("===== Stage 1: Data Ingestion Started =====")
            ensure_dir(self.cfg.stage_dir)
            ensure_dir(self.cfg.raw_data_dir)

            # 1) Download (kagglehub caches automatically)
            downloaded = kagglehub.dataset_download(self.cfg.dataset_name)
            logger.info("Dataset cache path: %s", downloaded)

            # 2) Extract if zip
            working = downloaded
            if self._is_zip(downloaded):
                working = self._extract_zip(
                    downloaded, os.path.join(self.cfg.raw_data_dir, "_tmp_extract")
                )

            # 3) Locate COCO root and stage required dirs
            coco_root = self._find_coco_root(working)
            train_name = Path(self.cfg.raw_train_image_dir).name
            val_name = Path(self.cfg.raw_val_image_dir).name

            self._copytree_if_missing(
                os.path.join(coco_root, train_name), self.cfg.raw_train_image_dir
            )
            src_val = os.path.join(coco_root, val_name)
            if os.path.exists(src_val):
                self._copytree_if_missing(src_val, self.cfg.raw_val_image_dir)
            else:
                logger.warning("Val images not found at %s — continuing", src_val)

            self._copytree_if_missing(
                os.path.join(coco_root, "annotations"), self.cfg.raw_annotation_dir
            )

            # 4) Validate critical files exist
            if not os.path.exists(self.cfg.raw_train_annotation_file):
                raise PipelineException(
                    f"Train annotation missing after staging: {self.cfg.raw_train_annotation_file}"
                )

            # 5) Write manifest
            manifest = {
                "dataset_name": self.cfg.dataset_name,
                "dataset_format": self.cfg.dataset_format,
                "created_at": datetime.now().isoformat(),
                "ingestion": {"mode": self.cfg.ingestion_mode},
                "paths": {
                    "train_images": self.cfg.raw_train_image_dir,
                    "val_images": self.cfg.raw_val_image_dir,
                },
                "annotations": {
                    "train": self.cfg.raw_train_annotation_file,
                    "val": self.cfg.raw_val_annotation_file,
                },
            }
            write_yaml(self.cfg.manifest_file, manifest)
            logger.info("Manifest → %s", self.cfg.manifest_file)
            logger.info("===== Stage 1: Data Ingestion Complete =====")

            return DataIngestionArtifact(
                stage_dir=self.cfg.stage_dir,
                manifest_file=self.cfg.manifest_file,
                ingestion_mode=self.cfg.ingestion_mode,
                dataset_format=self.cfg.dataset_format,
            )
        except PipelineException:
            raise
        except Exception as e:
            raise PipelineException("Data ingestion failed", e)
