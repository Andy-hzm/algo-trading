"""
Download SHARADAR/TICKERS and store to S3.

    s3://{bucket}/{env}/sharadar/tickers.parquet

Replaces backfill_sectors.py — Sharadar TICKERS has sector, SIC, exchange,
category (domestic/ADR/ETF), currency, and active/inactive status.

Usage:
    python scripts/backfill_sharadar_tickers.py
    python scripts/backfill_sharadar_tickers.py --env algotrading/dev
"""

import argparse
import logging
import os
import sys
from io import BytesIO

import boto3
import nasdaqdatalink
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def download_tickers() -> pd.DataFrame:
    nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    logger.info("Downloading SHARADAR/TICKERS...")
    df = nasdaqdatalink.get_table("SHARADAR/TICKERS", paginate=True)
    df.columns = df.columns.str.lower()
    logger.info(f"  {len(df):,} tickers | "
                f"categories: {df['category'].value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    args = parser.parse_args()

    df = download_tickers()

    bucket = os.environ["S3_BUCKET"]
    key = f"{args.env}/sharadar/tickers.parquet"
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"Written {len(df):,} rows → s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
