"""
Constants for VisionLLM Interaction Analysis training pipeline

"""

# ==================================================
# GLOBAL PIPELINE CONSTANTS
# ==================================================
DATASET_NAME: str = "awsaf49/coco-2017-dataset"
ARTIFACTS_DIR: str = "artifacts"
TRAINING_PIPELINE_NAME: str = "visionllm_interaction_pipeline"


# ==================================================
# DATA INGESTION STAGE 
# ==================================================

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

# Ingestion strategy 
DATA_INGESTION_MODE: str = "register"


# ==================================================
# RAW COCO DATASET (DOWNLOADED FROM KAGGLE)
# ==================================================

# Root directory containing raw COCO dataset
RAW_DATA_DIR: str = "data/raw"

# COCO image directories 
RAW_COCO_TRAIN_IMAGE_DIR: str = "data/raw/train2017"
RAW_COCO_VAL_IMAGE_DIR: str = "data/raw/val2017"

# COCO annotation files
RAW_COCO_ANNOTATION_DIR: str = "data/raw/annotations"
RAW_COCO_TRAIN_ANN_FILE: str = (
    "data/raw/annotations/instances_train2017.json"
)
RAW_COCO_VAL_ANN_FILE: str = (
    "data/raw/annotations/instances_val2017.json"
)


# ==================================================
# DATA INGESTION ARTIFACTS
# ==================================================

# Artifact directory for data ingestion stage
DATA_INGESTION_ARTIFACT_DIR: str = (
    f"{ARTIFACTS_DIR}/{DATA_INGESTION_DIR_NAME}"
)

# Manifest written by ingestion and consumed by all downstream stages
DATA_INGESTION_MANIFEST_FILE: str = "data_manifest.yaml"
