import os
from datetime import datetime

from visionllm_interaction.constants.training_pipeline import (
    ARTIFACTS_DIR,
    TRAINING_PIPELINE_NAME,
    DATA_INGESTION_DIR_NAME,
    DATA_INGESTION_MODE,
    RAW_DATA_DIR,
    RAW_COCO_TRAIN_IMAGE_DIR,
    RAW_COCO_VAL_IMAGE_DIR,
    RAW_COCO_ANNOTATION_DIR,
    RAW_COCO_TRAIN_ANN_FILE,
    RAW_COCO_VAL_ANN_FILE,
    DATA_INGESTION_MANIFEST_FILE,
    DATASET_NAME
)

class TrainingPipelineConfig:
    """
    Global training pipeline configuration.
    Creates a timestamped artifact directory for each run.
    """

    def __init__(self, timestamp: datetime = datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")

        self.pipeline_name: str = TRAINING_PIPELINE_NAME
        self.artifact_name: str = ARTIFACTS_DIR
        self.artifact_dir: str = os.path.join(self.artifact_name, timestamp)

        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Configuration for Data Ingestion stage .

    
    - Point to RAW COCO dataset (images + annotations)
    - Define where ingestion artifacts (manifest) are written
    """

    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        
        self.dataset_name: str = DATASET_NAME
        self.ingestion_mode: str = DATA_INGESTION_MODE  

        # -----------------------------
        # Artifact directory for stage
        # -----------------------------
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            DATA_INGESTION_DIR_NAME,
        )

        # -----------------------------
        # RAW COCO dataset paths
        # -----------------------------
        self.raw_data_dir: str = RAW_DATA_DIR

        self.raw_train_image_dir: str = RAW_COCO_TRAIN_IMAGE_DIR
        self.raw_val_image_dir: str = RAW_COCO_VAL_IMAGE_DIR

        self.raw_annotation_dir: str = RAW_COCO_ANNOTATION_DIR
        self.raw_train_annotation_file: str = RAW_COCO_TRAIN_ANN_FILE
        self.raw_val_annotation_file: str = RAW_COCO_VAL_ANN_FILE

        # -----------------------------
        # Ingestion artifact output
        # -----------------------------
        self.manifest_file_path: str = os.path.join(
            self.data_ingestion_dir,
            DATA_INGESTION_MANIFEST_FILE,
        )
