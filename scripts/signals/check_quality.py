"""
CLI wrapper around algotrading.signals.check_quality.

Usage:
    python scripts/signals/check_quality.py
    python scripts/signals/check_quality.py --key algotrading/prod/signals/cch.parquet --signal CCH --freq monthly
"""

import argparse
import os
import sys
from io import BytesIO

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from algotrading.signals.check_quality import run_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",    default="algotrading/prod/signals/cch.parquet")
    parser.add_argument("--signal", default="CCH")
    parser.add_argument("--freq",   default="monthly", choices=["monthly", "daily", "weekly"])
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    print(f"Loading s3://{bucket}/{args.key} ...")
    obj = s3.get_object(Bucket=bucket, Key=args.key)
    df = pd.read_parquet(BytesIO(obj["Body"].read()))
    df["date"] = pd.to_datetime(df["date"])
    print(f"Loaded {len(df):,} rows.")
    run_checks(df, signal_col=args.signal, freq=args.freq)


if __name__ == "__main__":
    main()
