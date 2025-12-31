from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """
    Artifact produced by the Data Ingestion stage .

    - Points to the ingestion manifest file created.
    """

    data_ingestion_dir: str
    manifest_file_path: str
    ingestion_mode: str
