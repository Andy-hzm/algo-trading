"""
Fetch sector/industry from yfinance for all tickers and store to S3.

    s3://zimeng/algotrading/prod/dim/sectors.parquet

Columns: ticker, sector, industry, sic_code, sic_description, market_cap

Usage:
    python scripts/backfill_sectors.py
    python scripts/backfill_sectors.py --tickers AAPL MSFT
    python scripts/backfill_sectors.py --workers 20
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import boto3
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algotrading.infra.s3_client import S3Client
from algotrading.infra.polygon_client import PolygonClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


_rate_lock = __import__("threading").Lock()
_last_call_time = [0.0]
_MIN_INTERVAL = 1.0  # seconds between requests globally


def fetch_one(ticker: str) -> dict:
    import time
    # Global rate limiter: enforce minimum interval between any two requests
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.time() - _last_call_time[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_time[0] = time.time()

    for attempt in range(4):
        try:
            info = yf.Ticker(ticker).info
            return {
                "ticker": ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "sector_key": info.get("sectorKey"),
                "industry_key": info.get("industryKey"),
            }
        except Exception as e:
            if "Too Many Requests" in str(e) and attempt < 3:
                backoff = 10 * (2 ** attempt)
                logger.warning(f"{ticker} rate limited, retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                logger.warning(f"{ticker} failed: {e}")
                return {"ticker": ticker, "sector": None, "industry": None, "sector_key": None, "industry_key": None}
    return {"ticker": ticker, "sector": None, "industry": None, "sector_key": None, "industry_key": None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--env",     default="algotrading/prod", choices=["algotrading/prod", "algotrading/dev"])
    args = parser.parse_args()

    s3 = S3Client(prefix=args.env)

    if args.tickers:
        tickers = args.tickers
    else:
        dim = s3.read_dim()
        if dim.empty:
            logger.error("Dim table not found.")
            sys.exit(1)
        tickers = dim["ticker"].tolist()

    logger.info(f"Fetching sector data for {len(tickers)} tickers with {args.workers} workers...")

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_one, t): t for t in tickers}
        for i, future in enumerate(as_completed(futures)):
            rows.append(future.result())
            if (i + 1) % 500 == 0:
                logger.info(f"  {i + 1}/{len(tickers)}")

    df = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    key = f"{args.env}/dim/sectors.parquet"
    bucket = os.environ["S3_BUCKET"]
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    buf = BytesIO()
    df.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())

    filled = df["sector"].notna().sum()
    logger.info(f"Done — {filled}/{len(df)} tickers have sector data → s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
