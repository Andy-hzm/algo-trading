"""
Download Fama-French 3-factor + momentum (UMD) monthly data and store to S3.

    s3://{bucket}/{env}/dim/ff_factors.parquet

Columns: date, mkt_rf, smb, hml, rf, umd  (all in decimal, not percent)

Source: Kenneth French Data Library (direct download, no pandas-datareader)

Usage:
    python scripts/backfill_ff_factors.py
    python scripts/backfill_ff_factors.py --env algotrading/dev
    python scripts/backfill_ff_factors.py --start 2015-01-01
"""

import argparse
import io
import logging
import os
import sys
import urllib.request
import zipfile
from io import BytesIO

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
UMD_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"


def _parse_french_csv(content: str, col_names: list) -> pd.DataFrame:
    """Parse a Ken French CSV: find monthly data block, return DataFrame."""
    lines = content.splitlines()
    start = next(
        i for i, l in enumerate(lines)
        if l.strip()[:4].isdigit() and len(l.strip()[:4]) == 4
    )
    end = next(i for i, l in enumerate(lines[start:], start) if l.strip() == "")
    df = pd.read_csv(
        io.StringIO("\n".join(lines[start:end])),
        header=None,
        names=col_names,
    )
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m")
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    value_cols = [c for c in col_names if c != "date"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce") / 100
    return df.dropna()


def download_ff3() -> pd.DataFrame:
    """Download FF3 monthly factors (MKT-RF, SMB, HML, RF)."""
    logger.info(f"Downloading FF3 from {FF3_URL}...")
    with urllib.request.urlopen(FF3_URL) as resp:
        zf = zipfile.ZipFile(BytesIO(resp.read()))
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        content = zf.read(csv_name).decode("utf-8")
    return _parse_french_csv(content, ["date", "mkt_rf", "smb", "hml", "rf"])


def download_umd() -> pd.DataFrame:
    """Download UMD (momentum) monthly factor."""
    logger.info(f"Downloading UMD from {UMD_URL}...")
    with urllib.request.urlopen(UMD_URL) as resp:
        zf = zipfile.ZipFile(BytesIO(resp.read()))
        csv_name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        content = zf.read(csv_name).decode("utf-8")
    return _parse_french_csv(content, ["date", "umd"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   default="algotrading/prod",
                        choices=["algotrading/prod", "algotrading/dev"])
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()

    ff3 = download_ff3()
    umd = download_umd()
    ff = ff3.merge(umd, on="date", how="left")
    ff = ff[ff["date"] >= args.start].reset_index(drop=True)

    logger.info(f"  {len(ff)} monthly observations ({ff['date'].min().date()} → {ff['date'].max().date()})")
    logger.info(f"  UMD coverage: {ff['umd'].notna().sum()}/{len(ff)} months")

    bucket = os.environ["S3_BUCKET"]
    key = f"{args.env}/dim/ff_factors.parquet"
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    buf = BytesIO()
    ff.to_parquet(buf, index=False, compression="snappy")
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    logger.info(f"Written → s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
