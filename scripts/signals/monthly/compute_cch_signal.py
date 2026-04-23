"""
Build the CCH signal panel from Sharadar SEP + SF1 + DAILY.

Sources:
    prices / returns  →  Sharadar SEP        (close, month-end, 1998–present)
    market equity     →  Sharadar DAILY      (marketcap, month-end)
    CCH fundamentals  →  Sharadar SF1 ARQ    (assetsc / assets)

Output:
    s3://{bucket}/{env}/signals/cch.parquet
    Columns: ticker, date (month-end), close, ret, ME, CCH

CCH = (assetsc/assets)_t - (assetsc/assets)_{t-1}
Point-in-time: SF1 filing is available at datekey (no look-ahead).

Usage:
    python scripts/signals/monthly/compute_cch_signal.py
    python scripts/signals/monthly/compute_cch_signal.py --env algotrading/dev
"""

import argparse
import logging
import os
import sys
from io import BytesIO

import boto3
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algotrading.infra.s3_client import S3Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def build_monthly_prices(s3: S3Client) -> pd.DataFrame:
    """Load Sharadar SEP daily prices, resample to month-end close + return."""
    logger.info("Loading Sharadar SEP (equity prices)...")
    daily = s3.read_sharadar_sep()
    logger.info(f"  {len(daily):,} rows | {daily['ticker'].nunique():,} tickers")

    daily["month"] = daily["date"].dt.to_period("M")

    monthly = (
        daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"])
        .agg(close=("close", "last"))
        .reset_index()
    )
    monthly["date"] = monthly["month"].dt.to_timestamp("M")
    monthly = monthly.drop(columns="month").sort_values(["ticker", "date"])
    # Filter zero/negative close prices before computing returns
    monthly = monthly[monthly["close"] > 0]
    monthly["ret"] = monthly.groupby("ticker")["close"].pct_change()

    logger.info(f"Monthly price panel: {len(monthly):,} ticker-months")
    return monthly


def build_monthly_me(s3: S3Client) -> pd.DataFrame:
    """Load Sharadar DAILY, get month-end marketcap."""
    logger.info("Loading Sharadar DAILY (marketcap)...")
    daily = s3.read_sharadar_daily()
    logger.info(f"  {len(daily):,} rows | columns: {list(daily.columns)}")

    daily["date"] = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M")

    me = (
        daily.sort_values(["ticker", "date"])
        .groupby(["ticker", "month"])
        .agg(ME=("marketcap", "last"))
        .reset_index()
    )
    me["ME"] = me["ME"] * 1e6  # Sharadar marketcap is in millions of USD
    me["date"] = me["month"].dt.to_timestamp("M")
    me = me.drop(columns="month")
    logger.info(f"Monthly ME panel: {len(me):,} ticker-months")
    return me


def build_cch_ratios(s3: S3Client) -> pd.DataFrame:
    """Load SF1 ARQ, compute point-in-time CCH ratio."""
    logger.info("Loading Sharadar SF1...")
    sf1 = s3.read_sharadar_sf1()
    sf1 = sf1[["ticker", "datekey", "assetsc", "assets"]].copy()
    sf1 = sf1.dropna(subset=["datekey", "assetsc", "assets"])
    sf1 = sf1[sf1["assets"] > 0]
    sf1["datekey"] = pd.to_datetime(sf1["datekey"])
    sf1["ca_ratio"] = sf1["assetsc"] / sf1["assets"]
    sf1 = sf1.sort_values(["ticker", "datekey"]).reset_index(drop=True)
    sf1["CCH"] = sf1.groupby("ticker")["ca_ratio"].diff()
    logger.info(f"SF1 filings with CCH: {sf1['CCH'].notna().sum():,}")
    return sf1[["ticker", "datekey", "CCH"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    args = parser.parse_args()

    s3 = S3Client()

    monthly_prices = build_monthly_prices(s3)
    monthly_me = build_monthly_me(s3)
    sf1_ratios = build_cch_ratios(s3)

    # Merge prices + ME on ticker + month-end date
    panel = monthly_prices.merge(monthly_me, on=["ticker", "date"], how="left")
    logger.info(f"After price+ME merge: {len(panel):,} rows | ME coverage: {panel['ME'].notna().sum():,}")

    # Point-in-time as-of merge with SF1 (datekey <= month-end date)
    panel["date"] = panel["date"].astype("datetime64[ms]")
    panel = panel.sort_values("date").reset_index(drop=True)
    sf1_sorted = (
        sf1_ratios.rename(columns={"datekey": "date"})
        .assign(date=lambda x: x["date"].astype("datetime64[ms]"))
        .sort_values("date")
        .reset_index(drop=True)
    )
    panel = pd.merge_asof(
        panel,
        sf1_sorted,
        on="date",
        by="ticker",
        direction="backward",
    )
    logger.info(f"After SF1 merge: {len(panel):,} rows | CCH coverage: {panel['CCH'].notna().sum():,}")

    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(
        f"Final panel: {len(panel):,} ticker-months | "
        f"{panel['ticker'].nunique():,} tickers | "
        f"{panel['date'].min().date()} → {panel['date'].max().date()}"
    )

    bucket = os.environ["S3_BUCKET"]
    key = f"{args.env}/signals/cch.parquet"
    s3_client = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    buf = BytesIO()
    panel.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3_client.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"Written → s3://{bucket}/{key}")

    # Run quality checks on the freshly written panel
    logger.info("Running signal quality checks...")
    from algotrading.signals.check_quality import run_checks
    run_checks(panel, signal_col="CCH", freq="monthly")


if __name__ == "__main__":
    main()
