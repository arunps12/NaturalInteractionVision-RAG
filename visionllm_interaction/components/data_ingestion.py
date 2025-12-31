import os
import sys
import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import kagglehub

from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import DataIngestionConfig
from visionllm_interaction.entity.artifact_entity import DataIngestionArtifact

logger = get_logger(__name__)


class DataIngestion:
    """
    Data ingestion for COCO .

    What this component does:
    1) Download dataset via kagglehub (to kaggle cache path)
    2) Ensure raw COCO structure exists under config.raw_data_dir (data/raw)
       - If downloaded is a zip, extract it
       - If downloaded is a folder, copy required COCO dirs/files into data/raw
    3) Validate that expected raw paths exist
    4) Write artifacts/data_ingestion/data_manifest.yaml
    5) Return DataIngestionArtifact
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.config = data_ingestion_config
        except Exception as e:
            raise CustomException("Failed to initialize DataIngestion", e)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    @staticmethod
    def _is_zip(path: str) -> bool:
        p = Path(path)
        return p.is_file() and p.suffix.lower() == ".zip"

    @staticmethod
    def _safe_mkdir(dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)

    def _extract_zip(self, zip_path: str, extract_to: str) -> str:
        """
        Extract zip_path into extract_to and return extract_to.
        """
        try:
            self._safe_mkdir(extract_to)
            logger.info(f"Extracting zip: {zip_path} -> {extract_to}")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_to)
            return extract_to
        except Exception as e:
            raise CustomException("Failed to extract dataset zip", e)

    def _find_coco_root(self, base_dir: str) -> str:
        """
        Locate the COCO root directory under base_dir.

        Returns the path to the detected COCO root.
        """
        try:
            base = Path(base_dir)

            
            if (base / Path(self.config.raw_train_image_dir).name).exists() and (base / "annotations").exists():
                return str(base)

            
            candidates = [base] + [p for p in base.glob("*") if p.is_dir()] + [p for p in base.glob("*/*") if p.is_dir()]
            train_dirname = Path(self.config.raw_train_image_dir).name  # "train2017"

            for c in candidates:
                if (c / train_dirname).exists() and (c / "annotations").exists():
                    return str(c)

            raise CustomException(
                f"Could not locate COCO root under: {base_dir}. Expected folders like '{train_dirname}/' and 'annotations/'."
            )
        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed while locating COCO root directory", e)

    def _copytree_if_missing(self, src: str, dst: str) -> None:
        """
        Copy a directory tree if dst doesn't exist.
        """
        if os.path.exists(dst):
            logger.info(f"Raw path already exists (skip copy): {dst}")
            return
        import shutil
        logger.info(f"Copying: {src} -> {dst}")
        shutil.copytree(src, dst)

    def _copyfile_if_missing(self, src: str, dst: str) -> None:
        """
        Copy a file if dst doesn't exist.
        """
        if os.path.exists(dst):
            logger.info(f"Raw file already exists (skip copy): {dst}")
            return
        import shutil
        self._safe_mkdir(os.path.dirname(dst))
        logger.info(f"Copying: {src} -> {dst}")
        shutil.copy2(src, dst)

    def _prepare_raw_data_dir(self, downloaded_path: str) -> None:
        """
        Ensures config.raw_data_dir contains the expected COCO structure.
        - If downloaded_path is a zip, extract it first
        - Locate COCO root
        - Copy required dirs/files into raw_data_dir
        """
        try:
            self._safe_mkdir(self.config.raw_data_dir)

           
            working_dir = downloaded_path
            if self._is_zip(downloaded_path):
                tmp_dir = os.path.join(self.config.raw_data_dir, "_tmp_extract")
                working_dir = self._extract_zip(downloaded_path, tmp_dir)

            coco_root = self._find_coco_root(working_dir)
            logger.info(f"Detected COCO root: {coco_root}")

           
            train_dirname = Path(self.config.raw_train_image_dir).name  # train2017
            val_dirname = Path(self.config.raw_val_image_dir).name      # val2017

            src_train = os.path.join(coco_root, train_dirname)
            src_val = os.path.join(coco_root, val_dirname)
            src_ann_dir = os.path.join(coco_root, "annotations")

            dst_train = self.config.raw_train_image_dir
            dst_val = self.config.raw_val_image_dir
            dst_ann_dir = self.config.raw_annotation_dir

            # Copy directories to raw (if not already present)
            if not os.path.exists(src_train):
                raise CustomException(f"Missing expected folder in dataset: {src_train}")
            if not os.path.exists(src_ann_dir):
                raise CustomException(f"Missing expected folder in dataset: {src_ann_dir}")

            self._copytree_if_missing(src_train, dst_train)

            
            if os.path.exists(src_val):
                self._copytree_if_missing(src_val, dst_val)
            else:
                logger.warning(f"val images folder not found at: {src_val} (continuing)")

            self._copytree_if_missing(src_ann_dir, dst_ann_dir)

            # Copy annotation files
            if not os.path.exists(self.config.raw_train_annotation_file):
                raise CustomException(
                    f"Train annotation file not found: {self.config.raw_train_annotation_file}"
                )
            if not os.path.exists(self.config.raw_val_annotation_file):
                logger.warning(
                    f"Val annotation file not found: {self.config.raw_val_annotation_file} (continuing)"
                )

            logger.info("Raw COCO data is prepared under: %s", self.config.raw_data_dir)

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed to prepare raw COCO directory", e)

    def _validate_raw_paths(self) -> None:
        """
        Validate that required raw paths exist.
        """
        try:
            required_dirs = [
                self.config.raw_data_dir,
                self.config.raw_train_image_dir,
                self.config.raw_annotation_dir,
            ]
            required_files = [
                self.config.raw_train_annotation_file,
            ]

            for d in required_dirs:
                if not os.path.exists(d):
                    raise CustomException(f"Required directory does not exist: {d}")

            for f in required_files:
                if not os.path.exists(f):
                    raise CustomException(f"Required file does not exist: {f}")

           
            if not os.path.exists(self.config.raw_val_image_dir):
                logger.warning(f"Validation: val image dir missing: {self.config.raw_val_image_dir}")

            if not os.path.exists(self.config.raw_val_annotation_file):
                logger.warning(f"Validation: val annotation file missing: {self.config.raw_val_annotation_file}")

            logger.info("Raw path validation completed.")

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Failed to validate raw dataset paths", e)

    def _write_manifest(self) -> None:
        """
        Write a simple YAML manifest consumed by downstream stages.
        """
        try:
            self._safe_mkdir(self.config.data_ingestion_dir)

            manifest_path = self.config.manifest_file_path
            created_at = datetime.now().isoformat()

            
            yaml_text = f"""dataset:
  name: coco2017
  source: kagglehub
  created_at: "{created_at}"

ingestion:
  mode: "{self.config.ingestion_mode}"

raw:
  root_dir: "{self.config.raw_data_dir}"
  train_images_dir: "{self.config.raw_train_image_dir}"
  val_images_dir: "{self.config.raw_val_image_dir}"
  annotation_dir: "{self.config.raw_annotation_dir}"
  train_annotation_file: "{self.config.raw_train_annotation_file}"
  val_annotation_file: "{self.config.raw_val_annotation_file}"
"""

            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write(yaml_text)

            logger.info(f"Wrote data manifest: {manifest_path}")

        except Exception as e:
            raise CustomException("Failed to write data manifest file", e)

    # ----------------------------------------------------------------------
    # Main entry
    # ----------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Main ingestion entrypoint.
        """
        try:
            logger.info("===== Data Ingestion Started (COCO, register-only) =====")

            # 1) Download dataset (kagglehub cache path)
            downloaded_path = kagglehub.dataset_download(self.config.dataset_name)
            logger.info(f"Downloaded dataset to: {downloaded_path}")

            # 2) Prepare raw data directory (data/raw)
            self._prepare_raw_data_dir(downloaded_path)

            # 3) Validate raw paths
            self._validate_raw_paths()

            # 4) Write manifest into artifacts/data_ingestion/
            self._write_manifest()

            logger.info("===== Data Ingestion Completed Successfully =====")

            return DataIngestionArtifact(
                data_ingestion_dir=self.config.data_ingestion_dir,
                manifest_file_path=self.config.manifest_file_path,
                ingestion_mode=self.config.ingestion_mode,
            )

        except CustomException:
            raise
        except Exception as e:
            raise CustomException("Error in data ingestion pipeline", e)
