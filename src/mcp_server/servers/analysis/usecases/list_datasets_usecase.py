"""Use case for listing available datasets."""

import logging
from typing import Any, Dict

from ..infrastructure.datasets_registry import DATASETS

logger = logging.getLogger(__name__)


class ListDatasetsUseCase:
    """Use case for listing all available datasets."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.datasets = DATASETS

    def execute(self) -> Dict[str, Any]:
        """
        Returns a list of available datasets with their metadata.

        Returns:
            Dictionary with a "datasets" key containing list of dataset metadata
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: list_datasets")
        logger.info("=" * 60)

        datasets_list = []
        for ds_id, meta in self.datasets.items():
            # Get storage information
            storage = meta.get("storage")
            if storage:
                storage_kind = storage.get("kind", "unknown")

                # Build location_hint based on storage kind
                if storage_kind == "local_csv":
                    path = storage.get("path")
                    location_hint = str(path) if path else "local"
                elif storage_kind == "s3_csv":
                    bucket = storage.get("bucket", "")
                    key = storage.get("key", "")
                    # Truncate long keys for display
                    if len(key) > 50:
                        key_display = key[:47] + "..."
                    else:
                        key_display = key
                    location_hint = (
                        f"s3://{bucket}/{key_display}" if bucket and key else "s3"
                    )
                else:
                    location_hint = "unknown"
            else:
                # Backward compatibility: use path if available
                storage_kind = "local_csv"  # Assume local if no storage block
                location_hint = str(meta.get("path", "unknown"))

            dataset_info = {
                "id": ds_id,
                "description": meta.get("description", ""),
                "code_name": meta.get("code_name"),
                "storage_kind": storage_kind,
                "location_hint": location_hint,
            }
            datasets_list.append(dataset_info)

        result = {"datasets": datasets_list}

        logger.info(f"OUTPUT SUMMARY: {len(datasets_list)} datasets available")
        logger.debug(f"  - Dataset IDs: {[ds['id'] for ds in datasets_list]}")
        logger.info("=" * 60)

        return result
