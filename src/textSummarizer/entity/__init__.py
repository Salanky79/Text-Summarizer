from dataclasses import dataclass
from pathlib import Path

# automatically create the directory structure for the data ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path