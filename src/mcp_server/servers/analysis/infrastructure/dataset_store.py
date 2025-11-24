"""Data access layer for loading datasets from various storage backends."""

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..schema.output import DatasetSchemaOutput
from .utils import detect_and_convert_datetime_columns

logger = logging.getLogger(__name__)


class DatasetStore:
    """
    Abstraction layer for loading datasets from various storage backends.

    Supports:
    - local_csv: Local CSV files
    - s3_csv: CSV files stored on S3

    This class hides storage-specific logic from MCP tools.
    """

    def __init__(self, datasets: Dict[str, Dict[str, Any]]):
        """
        Initialize the dataset store.

        Args:
            datasets: Dictionary mapping dataset IDs to their metadata
        """
        self._datasets = datasets

    def validate_id(self, dataset_id: str) -> Dict[str, Any]:
        """
        Validate that a dataset ID exists and return its metadata.

        Args:
            dataset_id: The dataset ID to validate

        Returns:
            The dataset metadata dictionary

        Raises:
            ValueError: If the dataset ID is not found
        """
        if dataset_id not in self._datasets:
            available_ids = ", ".join(self._datasets.keys())
            raise ValueError(
                f"Dataset '{dataset_id}' not found. "
                f"Available datasets: {available_ids}. "
                f"Use list_datasets() to see all available datasets."
            )
        return self._datasets[dataset_id]

    def load_dataframe(self, dataset_id: str) -> pd.DataFrame:
        """
        Load a dataset as a pandas DataFrame.

        Supports multiple storage kinds:
        - local_csv: Loads from local file system
        - s3_csv: Loads from S3 using pandas native S3 support

        Args:
            dataset_id: The dataset ID to load

        Returns:
            A pandas DataFrame with the dataset

        Raises:
            ValueError: If dataset_id is not found
            RuntimeError: If storage kind is not supported or loading fails
        """
        meta = self.validate_id(dataset_id)

        # Get storage configuration
        storage = meta.get("storage")
        if not storage:
            # Backward compatibility: check for top-level "path"
            if "path" in meta:
                logger.warning(
                    f"Dataset '{dataset_id}' uses legacy 'path' field. "
                    "Consider migrating to 'storage' block."
                )
                storage = {
                    "kind": "local_csv",
                    "path": meta["path"],
                    "read_params": {},
                }
            else:
                raise ValueError(
                    f"Dataset '{dataset_id}' has no 'storage' configuration or 'path' field."
                )

        storage_kind = storage.get("kind")
        logger.info(f"Loading dataset '{dataset_id}' from storage kind: {storage_kind}")

        if storage_kind == "local_csv":
            return self._load_local_csv(dataset_id, storage)
        elif storage_kind == "s3_csv":
            return self._load_s3_csv(dataset_id, storage)
        else:
            raise ValueError(
                f"Unsupported storage kind '{storage_kind}' for dataset '{dataset_id}'. "
                f"Supported kinds: 'local_csv', 's3_csv'"
            )

    def _load_local_csv(self, dataset_id: str, storage: Dict[str, Any]) -> pd.DataFrame:
        """
        Load a CSV file from local file system.

        Args:
            dataset_id: The dataset ID (for logging)
            storage: Storage configuration dict with 'path' and optional 'read_params'

        Returns:
            A pandas DataFrame
        """
        path = storage.get("path")
        if not path:
            raise ValueError(
                f"Dataset '{dataset_id}': 'path' is required for 'local_csv' storage"
            )

        # Convert to Path if it's a string
        if isinstance(path, str):
            path = Path(path)

        read_params = storage.get("read_params", {})

        logger.info(f"Loading local CSV: {path}")
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset '{dataset_id}': File not found at {path}. "
                "Check that the file exists and the path is correct."
            )

        try:
            df = pd.read_csv(path, **read_params)
            logger.info(
                f"Dataset '{dataset_id}' loaded: {len(df)} rows, {len(df.columns)} columns"
            )
            return df
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_id}' from {path}: {str(e)}"
            ) from e

    def _load_s3_csv(self, dataset_id: str, storage: Dict[str, Any]) -> pd.DataFrame:
        """
        Load a CSV file from S3.

        Args:
            dataset_id: The dataset ID (for logging)
            storage: Storage configuration dict with 'bucket', 'key', optional 'region', 'read_params'

        Returns:
            A pandas DataFrame

        Note:
            Credentials should be provided via environment variables, IAM role, or AWS credentials file.
            This method does not handle credentials directly.
        """
        bucket = storage.get("bucket")
        key = storage.get("key")

        if not bucket or not key:
            raise ValueError(
                f"Dataset '{dataset_id}': 'bucket' and 'key' are required for 's3_csv' storage"
            )

        # Build S3 URI
        s3_uri = f"s3://{bucket}/{key}"

        # Log safe information (no credentials)
        logger.info(
            f"Loading S3 CSV: s3://{bucket}/{key[:50]}..."
            if len(key) > 50
            else f"s3://{bucket}/{key}"
        )

        read_params = storage.get("read_params", {})
        region = storage.get("region")

        try:
            # Try using s3fs (if available) for pandas native S3 support
            # pandas.read_csv with s3:// URI requires s3fs
            try:
                import s3fs

                # s3fs will use boto3 credentials from environment/IAM
                # If region is specified, pass it via storage_options
                if region:
                    df = pd.read_csv(
                        s3_uri,
                        storage_options={"client_kwargs": {"region_name": region}},
                        **read_params,
                    )
                else:
                    # Let s3fs use default credentials and region
                    df = pd.read_csv(s3_uri, **read_params)
            except ImportError:
                # Fallback: use boto3 directly to download and read
                logger.info("s3fs not available, using boto3 directly")
                from io import BytesIO

                import boto3

                s3_client = (
                    boto3.client("s3", region_name=region)
                    if region
                    else boto3.client("s3")
                )
                obj = s3_client.get_object(Bucket=bucket, Key=key)
                df = pd.read_csv(BytesIO(obj["Body"].read()), **read_params)

            logger.info(
                f"Dataset '{dataset_id}' loaded from S3: {len(df)} rows, {len(df.columns)} columns"
            )
            return df

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Dataset '{dataset_id}': S3 object not found at s3://{bucket}/{key}. "
                "Check that the bucket and key are correct."
            ) from e
        except Exception as e:
            error_msg = str(e)
            if "AccessDenied" in error_msg or "Forbidden" in error_msg:
                raise PermissionError(
                    f"Dataset '{dataset_id}': Access denied to s3://{bucket}/{key}. "
                    "Check IAM permissions or AWS credentials."
                ) from e
            raise RuntimeError(
                f"Failed to load dataset '{dataset_id}' from S3 (s3://{bucket}/{key}): {str(e)}"
            ) from e

    def load_sample_dataframe(self, dataset_id: str, n: int = 5) -> pd.DataFrame:
        """
        Load a sample of a dataset (first n rows) for schema inspection.

        For efficiency, this may use nrows parameter when possible.

        Args:
            dataset_id: The dataset ID to load
            n: Number of rows to load (default: 5)

        Returns:
            A pandas DataFrame with at most n rows
        """
        meta = self.validate_id(dataset_id)
        storage = meta.get("storage")

        if not storage:
            # Backward compatibility
            if "path" in meta:
                storage = {
                    "kind": "local_csv",
                    "path": meta["path"],
                    "read_params": {},
                }
            else:
                raise ValueError(
                    f"Dataset '{dataset_id}' has no 'storage' configuration"
                )

        storage_kind = storage.get("kind")
        read_params = storage.get("read_params", {}).copy()

        # Add nrows for efficiency (only for CSV files)
        if storage_kind in ("local_csv", "s3_csv"):
            read_params["nrows"] = n

        # Temporarily modify storage to use nrows
        temp_storage = storage.copy()
        temp_storage["read_params"] = read_params

        if storage_kind == "local_csv":
            return self._load_local_csv(dataset_id, temp_storage)
        elif storage_kind == "s3_csv":
            return self._load_s3_csv(dataset_id, temp_storage)
        else:
            # Fallback: load full dataset and take head
            df = self.load_dataframe(dataset_id)
            return df.head(n)

    def get_schema(self, dataset_id: str) -> DatasetSchemaOutput:
        """
        Get schema information for a dataset.

        This loads the dataset, detects datetime columns, and returns
        schema information including columns, dtypes, sample rows, and row count.

        Args:
            dataset_id: The dataset ID to get schema for

        Returns:
            DatasetSchemaOutput with schema information
        """
        meta = self.validate_id(dataset_id)

        # Load full dataset for row count, but use sample for preview
        logger.info(f"Loading dataset '{dataset_id}' for schema inspection")
        df_full = self.load_dataframe(dataset_id)

        # Apply datetime conversion
        df_full = detect_and_convert_datetime_columns(df_full)

        # Get column information
        columns = df_full.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df_full.dtypes.items()}

        # Get sample rows (first 5)
        sample_df = df_full.head(5)
        sample_rows = sample_df.to_dict(orient="records")

        # Convert date-like columns and NaN values to strings/None for JSON serialization
        for row in sample_rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    row[key] = str(value)
                elif isinstance(value, (int, float)) and pd.isna(value):
                    row[key] = None

        return DatasetSchemaOutput(
            columns=columns,
            dtypes=dtypes,
            sample_rows=sample_rows,
            row_count=len(df_full),
            description=meta.get("description", ""),
        )
