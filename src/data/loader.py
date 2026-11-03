"""Dataset loading interfaces for XMCA-Net.

NOTE: Real data ingestion and sha256 checksum verification are staged for Task 1.
Synthetic data, dummy datasets, and fallback fall-throughs are strictly prohibited.
"""

from pathlib import Path
from typing import List, Dict, Any
from src.data.schema import validate_sample_schema


class DatasetLoader:
    """Base dataset loader enforcing strict file presence and schema checks."""

    def __init__(self, dataset_name: str, raw_file_path: Path):
        self.dataset_name = dataset_name
        self.raw_file_path = Path(raw_file_path)

    def load_samples(self, tier: int) -> List[Dict[str, Any]]:
        """Load and validate dataset samples for a given progressive tier.

        Raises:
            FileNotFoundError: If the verified raw data file does not exist.
            NotImplementedError: If parsing logic is pending implementation.
        """
        if not self.raw_file_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {self.raw_file_path}. "
                "Synthetic data and dummy fallbacks are strictly prohibited under strict mode. "
                "Data download must be executed in accordance with docs/DATASET_PROTOCOL.md."
            )

        raise NotImplementedError(
            f"Dataset loader for {self.dataset_name} is pending Task 1 implementation. "
            "Placeholder implementations returning dummy samples are strictly prohibited."
        )
