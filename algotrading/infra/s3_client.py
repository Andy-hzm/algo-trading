import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from typing import List

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class S3Client:
    """
    Read/write Parquet files on S3.

    Layout:
        s3://{bucket}/{prefix}/
        ├── dim/tickers.parquet
        └── bars/hourly/ticker={TICKER}/year={YYYY}/month={MM}/data.parquet

    Partitioned by ticker then time — each ticker owns its own files so
    parallel writes never conflict. Batched writes reduce S3 PUT count.

    Use prefix='prod' for real data, prefix='dev' for experiments.
    """

    def __init__(self, bucket: str = None, region: str = None, prefix: str = "prod"):
        self.bucket = bucket or os.environ["S3_BUCKET"]
        self.prefix = prefix
        self._s3 = boto3.client(
            "s3",
            region_name=region or os.environ.get("AWS_REGION", "us-east-1"),
        )

    @property
    def _dim_key(self):
        return f"{self.prefix}/dim/tickers.parquet"

    @property
    def _bars_prefix(self):
        return f"{self.prefix}/bars/hourly"

    # ------------------------------------------------------------------
    # Dim table
    # ------------------------------------------------------------------

    def write_dim(self, df: pd.DataFrame) -> None:
        """Write ticker dim table to s3://{bucket}/{prefix}/dim/tickers.parquet"""
        self._write_parquet(df, self._dim_key)
        logger.info(f"Wrote dim table ({len(df)} rows) → s3://{self.bucket}/{self._dim_key}")

    def read_dim(self) -> pd.DataFrame:
        """Read ticker dim table from S3. Returns empty DataFrame if not found."""
        return self._read_parquet(self._dim_key)

    # ------------------------------------------------------------------
    # Bars — write
    # ------------------------------------------------------------------

    def write_bars_batch(self, df: pd.DataFrame, max_workers: int = 20) -> None:
        """
        Write a batch of OHLCV bars (multiple tickers) to S3 in parallel.

        Layout: ticker={TICKER}/year={YYYY}/month={MM}/data.parquet
        Each ticker writes to its own files — no conflicts between parallel workers.
        S3 PUTs are issued concurrently via ThreadPoolExecutor.

        Expects df indexed by UTC timestamp with a 'ticker' column.
        Always overwrites — caller is responsible for dedup if needed.
        """
        if df.empty:
            return

        df = df.copy()
        df["_year"] = df.index.year
        df["_month"] = df.index.month

        # Build list of (key, chunk) pairs to write
        writes = []
        for (ticker, year, month), chunk in df.groupby(["ticker", "_year", "_month"]):
            key = f"{self._bars_prefix}/ticker={ticker}/year={year}/month={month:02d}/data.parquet"
            writes.append((key, chunk.drop(columns=["_year", "_month"])))

        # Issue all S3 PUTs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._write_parquet, chunk, key): key for key, chunk in writes}
            for future in as_completed(futures):
                future.result()  # raise if any write failed

        tickers = df["ticker"].nunique()
        logger.info(f"Wrote {len(df)} rows ({tickers} tickers, {len(writes)} partitions) → {self._bars_prefix}/")

    # ------------------------------------------------------------------
    # Bars — read
    # ------------------------------------------------------------------

    def read_bars(self, start: str, end: str, tickers: List[str] = None) -> pd.DataFrame:
        """
        Read bars for a date range.

        Args:
            start:   'YYYY-MM-DD'
            end:     'YYYY-MM-DD'
            tickers: list of ticker symbols to load. If None, loads all tickers
                     (slower — scans all ticker partitions).

        Returns a DataFrame indexed by UTC timestamp.
        """
        start_dt = pd.Timestamp(start, tz="UTC")
        end_dt = pd.Timestamp(end, tz="UTC")

        if tickers is None:
            tickers = self._list_tickers()

        keys = []
        for ticker in tickers:
            keys.extend(self._ticker_partition_keys(ticker, start_dt, end_dt))

        frames = []
        for key in keys:
            chunk = self._read_parquet(key)
            if not chunk.empty:
                frames.append(chunk)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames).sort_index()
        return df.loc[start_dt:end_dt]

    def read_ticker(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Convenience method to read a single ticker."""
        return self.read_bars(start, end, tickers=[ticker])

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

    def _ticker_partition_keys(self, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> list:
        keys = []
        current = start.replace(day=1)
        while current <= end:
            key = f"{self._bars_prefix}/ticker={ticker}/year={current.year}/month={current.month:02d}/data.parquet"
            keys.append(key)
            current += pd.DateOffset(months=1)
        return keys

    def _list_tickers(self) -> list:
        """List all ticker prefixes available in S3."""
        prefix = f"{self._bars_prefix}/ticker="
        paginator = self._s3.get_paginator("list_objects_v2")
        tickers = set()
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix, Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                # cp['Prefix'] = 'prod/bars/hourly/ticker=AAPL/'
                ticker = cp["Prefix"].rstrip("/").split("ticker=")[-1]
                tickers.add(ticker)
        return sorted(tickers)
