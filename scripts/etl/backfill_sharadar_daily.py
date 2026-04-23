"""
Download SHARADAR/DAILY (prices + volume) and store to S3 partitioned by year+month.

    s3://{bucket}/{env}/sharadar/daily/year={Y}/month={MM}/data.parquet

Downloads all (year, month) pairs in parallel — 132 partitions for 2016-2026.

Usage:
    python scripts/backfill_sharadar_daily.py
    python scripts/backfill_sharadar_daily.py --start 2016 --end 2026
    python scripts/backfill_sharadar_daily.py --workers 12
    python scripts/backfill_sharadar_daily.py --env algotrading/dev
"""

import argparse
import calendar
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


def download_and_store(year: int, month: int, env: str, bucket: str, s3_client) -> None:
    last_day = calendar.monthrange(year, month)[1]
    gte = f"{year}-{month:02d}-01"
    lte = f"{year}-{month:02d}-{last_day:02d}"

    df = nasdaqdatalink.get_table(
        "SHARADAR/DAILY",
        date={"gte": gte, "lte": lte},
        paginate=True,
    )
    df.columns = df.columns.str.lower()
    if df.empty:
        logger.info(f"  {year}-{month:02d}: no data, skipping.")
        return
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(f"  {year}-{month:02d}: {len(df):,} rows | {df['ticker'].nunique():,} tickers")

    key = f"{env}/sharadar/daily/year={year}/month={month:02d}/data.parquet"
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"  {year}-{month:02d}: written → s3://{bucket}/{key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    parser.add_argument("--start",   type=int, default=1997)
    parser.add_argument("--end",     type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    partitions = [
        (y, m)
        for y in range(args.start, args.end + 1)
        for m in range(1, 13)
    ]
    logger.info(f"Downloading {len(partitions)} month-partitions with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_and_store, y, m, args.env, bucket, s3): (y, m)
            for y, m in partitions
        }
        for future in as_completed(futures):
            future.result()

    logger.info("Done.")


if __name__ == "__main__":
    main()
