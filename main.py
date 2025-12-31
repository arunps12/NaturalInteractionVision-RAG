from visionllm_interaction.logger.logger import get_logger
from visionllm_interaction.exception.custom_exception import CustomException
from visionllm_interaction.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
)
from visionllm_interaction.components.data_ingestion import DataIngestion

logger = get_logger(__name__)


def main():
    """
    Entry point for VisionLLM Interaction Analysis pipeline.

    """
    try:
        logger.info("=== VisionLLM Interaction Analysis: Pipeline Started ===")

        # ------------------------------------------------------------
        # 1) Build training pipeline config (timestamped artifacts)
        # ------------------------------------------------------------
        training_pipeline_config = TrainingPipelineConfig()
        logger.info(f"Run artifacts directory: {training_pipeline_config.artifact_dir}")

        # ------------------------------------------------------------
        # 2) Data Ingestion
        # ------------------------------------------------------------
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)

        logger.info("Starting data ingestion stage...")
        logger.info(f"Dataset: {data_ingestion_config.dataset_name}")
        logger.info(f"Ingestion mode: {data_ingestion_config.ingestion_mode}")
        logger.info(f"Dataset format: {data_ingestion_config.dataset_format}")

        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info("Data ingestion stage completed successfully.")
        logger.info(f"Manifest written to: {data_ingestion_artifact.manifest_file_path}")
        logger.info(f"DataIngestionArtifact: {data_ingestion_artifact}")

        logger.info("=== VisionLLM Interaction Analysis: Pipeline Finished ===")

    except CustomException as ce:
        logger.error(f"Pipeline failed with CustomException: {ce}")
        raise

    except Exception as e:
        logger.error("Pipeline failed with an unexpected exception.")
        raise CustomException("Unexpected error in main pipeline execution", e)


if __name__ == "__main__":
    main()
