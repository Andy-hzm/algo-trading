"""
Download SHARADAR metadata tables: SP500, ACTIONS, EVENTS.

    s3://{bucket}/{env}/sharadar/sp500.parquet      — S&P500 membership history
    s3://{bucket}/{env}/sharadar/actions.parquet    — splits, dividends, delistings
    s3://{bucket}/{env}/sharadar/events.parquet     — earnings dates, guidance

Run after backfill_sharadar_tickers.py and backfill_sharadar_sf1.py.

Usage:
    python scripts/backfill_sharadar_meta.py
    python scripts/backfill_sharadar_meta.py --tables sp500 actions
"""

import argparse
import logging
import os
import sys
import time
import zipfile
from io import BytesIO

import boto3
import nasdaqdatalink
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TABLES = {
    "sp500":   ("SHARADAR/SP500",   "sp500.parquet"),
    "actions": ("SHARADAR/ACTIONS", "actions.parquet"),
    "events":  ("SHARADAR/EVENTS",  "events.parquet"),
}


def bulk_export_download(nasdaql_key: str, api_key: str) -> pd.DataFrame:
    """Use Nasdaq bulk export API for tables too large for get_table()."""
    export_url = (
        f"https://data.nasdaq.com/api/v3/datatables/{nasdaql_key}"
        f"?qopts.export=true&api_key={api_key}"
    )
    # Request the export — Nasdaq queues it and returns a download link
    resp = requests.get(export_url)
    resp.raise_for_status()
    info = resp.json()
    status = info["datatable_bulk_download"]["file"]["status"]
    download_link = info["datatable_bulk_download"]["file"]["link"]

    # Wait for export to be ready
    while status == "creating":
        logger.info("  Export still generating, waiting 5s...")
        time.sleep(5)
        resp = requests.get(export_url)
        resp.raise_for_status()
        info = resp.json()
        status = info["datatable_bulk_download"]["file"]["status"]
        download_link = info["datatable_bulk_download"]["file"]["link"]

    logger.info(f"  Downloading bulk export from {download_link[:60]}...")
    data = requests.get(download_link)
    data.raise_for_status()
    with zipfile.ZipFile(BytesIO(data.content)) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            return pd.read_csv(f, low_memory=False)


def download_and_store(table_name: str, nasdaql_key: str, s3_key: str,
                       bucket: str, s3_client, api_key: str) -> None:
    logger.info(f"Downloading {nasdaql_key}...")
    try:
        df = nasdaqdatalink.get_table(nasdaql_key, paginate=True)
    except nasdaqdatalink.errors.data_link_error.LimitExceededError:
        logger.info(f"  Too large for get_table(), switching to bulk export...")
        df = bulk_export_download(nasdaql_key, api_key)

    df.columns = df.columns.str.lower()
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    logger.info(f"  {len(df):,} rows")
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=s3_key, Body=buf.read())
    logger.info(f"  Written → s3://{bucket}/{s3_key}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    parser.add_argument("--tables", nargs="+", default=list(TABLES.keys()),
                        choices=list(TABLES.keys()))
    args = parser.parse_args()

    nasdaqdatalink.ApiConfig.api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

    api_key = os.environ["NASDAQ_DATA_LINK_API_KEY"]
    for name in args.tables:
        nasdaql_key, filename = TABLES[name]
        s3_key = f"{args.env}/sharadar/{filename}"
        download_and_store(name, nasdaql_key, s3_key, bucket, s3, api_key)


if __name__ == "__main__":
    main()
