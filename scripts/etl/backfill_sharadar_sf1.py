"""
Download SHARADAR/SF1 (Core US Fundamentals, ARQ dimension) and store to S3.

    s3://{bucket}/{env}/sharadar/sf1.parquet

Two download modes:
  default    — parallel chunks by calendardate range (fast, skips empty ranges)
  --no-chunk — single bulk download with no filter (use in dev to diagnose coverage)

Usage:
    python scripts/backfill_sharadar_sf1.py
    python scripts/backfill_sharadar_sf1.py --workers 8
    python scripts/backfill_sharadar_sf1.py --no-chunk --env algotrading/dev
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import boto3
import nasdaqdatalink
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# One chunk per year — each worker downloads one full year in parallel
CHUNK_YEARS = 1
START_YEAR  = 1990
END_YEAR    = 2026


def date_chunks(start: int, end: int, chunk_years: int) -> list[tuple[str, str]]:
    """Split [start, end] into (gte, lte) date string pairs."""
    chunks = []
    year = start
    while year <= end:
        chunk_end = min(year + chunk_years - 1, end)
        chunks.append((f"{year}-01-01", f"{chunk_end}-12-31"))
        year += chunk_years
    return chunks


def download_chunk(gte: str, lte: str) -> pd.DataFrame:
    df = nasdaqdatalink.get_table(
        "SHARADAR/SF1",
        dimension="ARQ",
        calendardate={"gte": gte, "lte": lte},
        paginate=True,
    )
    df.columns = df.columns.str.lower()
    logger.info(f"  {gte[:4]}–{lte[:4]}: {len(df):,} rows")
    return df


def download_sf1(workers: int = 6) -> pd.DataFrame:
    nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    chunks = date_chunks(START_YEAR, END_YEAR, CHUNK_YEARS)
    logger.info(f"Downloading SF1 in {len(chunks)} chunks with {workers} workers...")

    frames = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_chunk, gte, lte): (gte, lte)
                   for gte, lte in chunks}
        for future in as_completed(futures):
            frames.append(future.result())

    df = pd.concat(frames, ignore_index=True)
    df["datekey"]      = pd.to_datetime(df["datekey"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    df = df.sort_values(["ticker", "datekey"]).reset_index(drop=True)
    logger.info(f"Total: {len(df):,} rows | {df['ticker'].nunique():,} tickers | "
                f"{df['datekey'].min().date()} → {df['datekey'].max().date()}")
    return df


def download_sf1_bulk() -> pd.DataFrame:
    """Download the full SF1 ARQ table with no date filter — for diagnosing coverage."""
    nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    logger.info("Downloading SF1 (bulk, no date filter)...")
    df = nasdaqdatalink.get_table("SHARADAR/SF1", dimension="ARQ", paginate=True)
    df.columns = df.columns.str.lower()
    df["datekey"]      = pd.to_datetime(df["datekey"])
    df["calendardate"] = pd.to_datetime(df["calendardate"])
    df = df.sort_values(["ticker", "datekey"]).reset_index(drop=True)
    logger.info(f"Total: {len(df):,} rows | {df['ticker'].nunique():,} tickers | "
                f"{df['datekey'].min().date()} → {df['datekey'].max().date()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    parser.add_argument("--workers",  type=int, default=6)
    parser.add_argument("--no-chunk", action="store_true",
                        help="Download full table at once (use with --env algotrading/dev to test coverage)")
    args = parser.parse_args()

    df = download_sf1_bulk() if args.no_chunk else download_sf1(workers=args.workers)

    bucket = os.environ["S3_BUCKET"]
    key = f"{args.env}/sharadar/sf1.parquet"
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"Written {len(df):,} rows → s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
