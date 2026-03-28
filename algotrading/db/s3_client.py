import os
import logging
from io import BytesIO

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Client:
    """
    Read/write Parquet files on S3.

    Layout:
        s3://{bucket}/
        ├── dim/tickers.parquet                          ← ticker universe
        └── bars/hourly/year={YYYY}/month={MM}/data.parquet  ← OHLCV bars
    """

    def __init__(self, bucket: str = None, region: str = None):
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self._s3 = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        )

    @property
    def _dim_key(self):
        return "dim/tickers.parquet"

    @property
    def _bars_prefix(self):
        return "bars/hourly"

    # ------------------------------------------------------------------
    # Dim table
    # ------------------------------------------------------------------

    def write_dim(self, df: pd.DataFrame) -> None:
        """Write ticker dim table to s3://{bucket}/{env}/dim/tickers.parquet"""
        self._write_parquet(df, self._dim_key)
        logger.info(f"Wrote dim table ({len(df)} rows) → s3://{self.bucket}/{self._dim_key}")

    def read_dim(self) -> pd.DataFrame:
        """Read ticker dim table from S3. Returns empty DataFrame if not found."""
        return self._read_parquet(self._dim_key)

    # ------------------------------------------------------------------
    # Bars
    # ------------------------------------------------------------------

    def write_bars(self, df: pd.DataFrame) -> None:
        """
        Write OHLCV bars to S3, partitioned by year/month.

        Expects df indexed by UTC timestamp with a 'ticker' column.
        Splits into monthly partitions and writes each separately.
        """
        if df.empty:
            return

        df = df.copy()
        df["year"] = df.index.year
        df["month"] = df.index.month

        for (year, month), chunk in df.groupby(["year", "month"]):
            key = f"{self._bars_prefix}/year={year}/month={month:02d}/data.parquet"
            # If partition already exists, merge with existing data
            existing = self._read_parquet(key)
            if not existing.empty:
                chunk = chunk.drop(columns=["year", "month"])
                combined = pd.concat([existing, chunk])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
            else:
                chunk = chunk.drop(columns=["year", "month"])
                combined = chunk

            self._write_parquet(combined, key)
            logger.info(f"Wrote {len(combined)} rows → s3://{self.bucket}/{key}")

    def read_bars(self, start: str, end: str) -> pd.DataFrame:
        """
        Read bars for a date range by loading only the relevant partitions.

        Args:
            start: 'YYYY-MM-DD'
            end:   'YYYY-MM-DD'

        Returns a DataFrame indexed by UTC timestamp.
        """
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")

        keys = self._partition_keys(start_dt, end_dt)
        frames = []
        for key in keys:
            df = self._read_parquet(key)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        return df.loc[start_dt:end_dt]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_parquet(self, df: pd.DataFrame, key: str) -> None:
        buf = BytesIO()
        df.to_parquet(buf, index=True, compression="snappy")
        buf.seek(0)
        self._s3.put_object(Bucket=self.bucket, Key=key, Body=buf.read())

    def _read_parquet(self, key: str) -> pd.DataFrame:
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=key)
            buf = BytesIO(obj["Body"].read())
            return pd.read_parquet(buf)
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return pd.DataFrame()
            raise

    def _partition_keys(self, start: pd.Timestamp, end: pd.Timestamp) -> list:
        """Generate all year/month partition keys between start and end."""
        keys = []
        current = start.replace(day=1)
        while current <= end:
            key = f"{self._bars_prefix}/year={current.year}/month={current.month:02d}/data.parquet"
            keys.append(key)
            current += pd.DateOffset(months=1)
        return keys
