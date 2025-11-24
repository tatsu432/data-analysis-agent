"""Use case for getting dataset schema information."""

import logging

from ..infrastructure.dataset_store import DatasetStore
from ..infrastructure.datasets_registry import DATASETS
from ..schema.input import GetDatasetSchemaInput
from ..schema.output import DatasetSchemaOutput

logger = logging.getLogger(__name__)


class GetDatasetSchemaUseCase:
    """Use case for getting dataset schema information."""

    def __init__(self) -> None:
        """Initialize the use case."""
        self.dataset_store = DatasetStore(DATASETS)

    def execute(self, payload: GetDatasetSchemaInput) -> DatasetSchemaOutput:
        """
        Returns schema information for the specified dataset.

        Args:
            payload: Input containing dataset_id

        Returns:
            DatasetSchemaOutput with columns, dtypes, sample rows, and row count

        Raises:
            ValueError: If dataset_id is not found in the registry
        """
        logger.info("=" * 60)
        logger.info("USECASE EXECUTION: get_dataset_schema")
        logger.info("=" * 60)
        logger.info(f"INPUT - dataset_id: {payload.dataset_id}")

        # Use DatasetStore to get schema (handles validation and loading)
        try:
            result = self.dataset_store.get_schema(payload.dataset_id)

            logger.info("OUTPUT SUMMARY:")
            logger.info(f"  - Columns: {len(result.columns)}")
            logger.info(f"  - Row count: {result.row_count}")
            logger.info(f"  - Sample rows: {len(result.sample_rows)}")
            logger.debug(f"  - Column names: {result.columns}")
            logger.info("=" * 60)

            return result
        except ValueError as e:
            # Re-raise ValueError as-is (already has good error message)
            logger.error(str(e))
            raise
        except Exception as e:
            # Wrap other exceptions with context
            error_msg = (
                f"Failed to get schema for dataset '{payload.dataset_id}': {str(e)}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
