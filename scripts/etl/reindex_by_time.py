"""
Copy hourly layout → hourly_by_time layout.

    hourly/ticker=X/year=Y/month=M/data.parquet
        →
    hourly_by_time/year=Y/month=M/ticker=X/data.parquet

Usage:
    python scripts/reindex_by_time.py                              # full prod
    python scripts/reindex_by_time.py --env dev                    # dev env
    python scripts/reindex_by_time.py --tickers AAPL MSFT          # specific tickers
    python scripts/reindex_by_time.py --year 2020 --month 01       # specific time range
    python scripts/reindex_by_time.py --tickers AAPL --year 2020   # combined
"""

import argparse
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

KEY_PATTERN = re.compile(r"ticker=(?P<ticker>[^/]+)/year=(?P<year>[^/]+)/month=(?P<month>[^/]+)/data\.parquet$")


def list_source_keys(s3, bucket: str, src_prefix: str, tickers: list[str] | None, year: str | None, month: str | None) -> list[str]:
    paginator = s3.get_paginator("list_objects_v2")
    keys = []

    # Narrow the S3 prefix as much as possible to avoid listing everything
    if tickers and len(tickers) == 1:
        ticker_prefix = f"{src_prefix}/ticker={tickers[0]}"
        if year:
            ticker_prefix += f"/year={year}"
            if month:
                ticker_prefix += f"/month={month}"
        prefixes = [ticker_prefix]
    else:
        prefixes = [src_prefix]

    for prefix in prefixes:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix + "/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                m = KEY_PATTERN.search(key)
                if not m:
                    continue
                if tickers and m["ticker"] not in tickers:
                    continue
                if year and m["year"] != year:
                    continue
                if month and m["month"] != month:
                    continue
                keys.append(key)

    return keys


def copy_one(s3, bucket: str, src: str, dst_prefix: str) -> str:
    m = KEY_PATTERN.search(src)
    dst = f"{dst_prefix}/year={m['year']}/month={m['month']}/ticker={m['ticker']}/data.parquet"
    s3.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src},
        Key=dst,
    )
    return dst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",      default="prod", choices=["prod", "dev"])
    parser.add_argument("--tickers",  nargs="+", default=None)
    parser.add_argument("--year",     default=None)
    parser.add_argument("--month",    default=None)
    parser.add_argument("--workers",  type=int, default=20)
    args = parser.parse_args()

    src_prefix = f"algotrading/{args.env}/bars/hourly"
    dst_prefix = f"algotrading/{args.env}/bars/hourly_by_time"

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    logger.info(f"Listing source keys (env={args.env}, tickers={args.tickers}, year={args.year}, month={args.month})...")
    keys = list_source_keys(s3, bucket, src_prefix, args.tickers, args.year, args.month)
    logger.info(f"Found {len(keys)} files to copy")

    if not keys:
        logger.info("Nothing to do.")
        return

    done = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(copy_one, s3, bucket, src, dst_prefix): src for src in keys}
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
                if done % 100 == 0:
                    logger.info(f"  {done}/{len(keys)}")
            except Exception as e:
                errors += 1
                logger.error(f"Failed {futures[future]}: {e}")

    logger.info(f"Done — {done} copied, {errors} errors")


if __name__ == "__main__":
    main()
