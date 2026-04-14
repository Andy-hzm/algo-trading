"""
Resample hourly bars → daily bars and write to S3.

Reads from:  {env}/bars/hourly_by_time/year={Y}/month={M}/ticker={T}/data.parquet
Writes to:   {env}/bars/daily/year={Y}/month={M}/data.parquet

Aggregation:
    open   = first
    high   = max
    low    = min
    close  = last
    volume = sum
    vwap   = sum(vwap * volume) / sum(volume)

Usage:
    python scripts/resample_daily.py
    python scripts/resample_daily.py --env algotrading/dev
    python scripts/resample_daily.py --year 2020 --month 01
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def list_months(s3, bucket: str, src_prefix: str) -> list[tuple[str, str]]:
    """Return (year, month) pairs available under src_prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    months = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=src_prefix + "/", Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # cp = 'env/bars/hourly_by_time/year=2020/'
            year = cp["Prefix"].rstrip("/").split("year=")[-1]
            for page2 in paginator.paginate(Bucket=bucket, Prefix=cp["Prefix"], Delimiter="/"):
                for cp2 in page2.get("CommonPrefixes", []):
                    month = cp2["Prefix"].rstrip("/").split("month=")[-1]
                    months.add((year, month))
    return sorted(months)


def read_parquet(s3, bucket: str, key: str) -> pd.DataFrame:
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_parquet(BytesIO(obj["Body"].read()))
    except s3.exceptions.NoSuchKey:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def resample_month(s3, bucket: str, src_prefix: str, dst_prefix: str, year: str, month: str) -> int:
    """Read all ticker files for year/month, resample to daily, write one file."""
    prefix = f"{src_prefix}/year={year}/month={month}/"
    paginator = s3.get_paginator("list_objects_v2")

    frames = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            ticker = key.split("ticker=")[-1].split("/")[0]
            df = read_parquet(s3, bucket, key)
            if df.empty:
                continue
            df["ticker"] = ticker
            frames.append(df)

    if not frames:
        return 0

    hourly = pd.concat(frames)
    hourly.index = pd.to_datetime(hourly.index, utc=True)
    hourly["vwap_vol"] = hourly["vwap"] * hourly["volume"]

    # Resample to daily
    daily = hourly.groupby("ticker").resample("1D").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        vwap_vol_sum=("vwap_vol", "sum"),
    ).dropna(subset=["close"]).reset_index()

    daily["vwap"] = daily["vwap_vol_sum"] / daily["volume"]
    daily = daily.drop(columns=["vwap_vol_sum"])

    daily = daily.sort_values(["timestamp", "ticker"])

    # Write
    dst_key = f"{dst_prefix}/year={year}/month={month}/data.parquet"
    buf = BytesIO()
    daily.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=dst_key, Body=buf.read())

    return len(daily)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        default="algotrading/prod", choices=["algotrading/prod", "algotrading/dev"])
    parser.add_argument("--year",       default=None)
    parser.add_argument("--month",      default=None)
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--end-year",   type=int, default=None)
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]
    src_prefix = f"{args.env}/bars/hourly_by_time"
    dst_prefix = f"{args.env}/bars/daily"

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    if args.year and args.month:
        months = [(args.year, args.month)]
    elif args.year:
        months = [(args.year, f"{m:02d}") for m in range(1, 13)]
    elif args.start_year or args.end_year:
        start = args.start_year or 2016
        end = args.end_year or 2026
        months = [(str(y), f"{m:02d}") for y in range(start, end + 1) for m in range(1, 13)]
    else:
        logger.info("Listing available year/month partitions...")
        months = list_months(s3, bucket, src_prefix)

    logger.info(f"Processing {len(months)} months → {dst_prefix}/")

    total_rows = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(resample_month, s3, bucket, src_prefix, dst_prefix, y, m): (y, m)
            for y, m in months
        }
        for future in as_completed(futures):
            y, m = futures[future]
            try:
                rows = future.result()
                total_rows += rows
                logger.info(f"  {y}-{m} → {rows} daily bars written")
            except Exception as e:
                logger.error(f"  {y}-{m} failed: {e}")

    logger.info(f"Done — {total_rows} total daily bars written")


if __name__ == "__main__":
    main()
