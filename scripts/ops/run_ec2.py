"""
Run jobs on EC2 via SSM. Code is synced via S3 (no GitHub/SSH needed).

Usage:
    python scripts/run_ec2.py                              # full backfill, prod
    python scripts/run_ec2.py --setup                      # first-time setup
    python scripts/run_ec2.py --no-stop                    # keep instance running after job
    python scripts/run_ec2.py --job resample-daily
    python scripts/run_ec2.py --job sharadar-sf1           # download SF1 fundamentals
    python scripts/run_ec2.py --job sharadar-daily         # download DAILY prices
    python scripts/run_ec2.py --job sharadar-tickers       # download TICKERS dim
    python scripts/run_ec2.py --job sharadar-meta          # download SP500/ACTIONS/EVENTS
    python scripts/run_ec2.py --job sharadar-all           # run all 4 sharadar jobs
"""

import argparse
import logging
import os
import subprocess
import sys
from io import BytesIO

import boto3
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algotrading.infra.ec2_runner import EC2Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_S3_KEY = "code/code.zip"


def upload_code(bucket: str) -> None:
    """Zip local code and upload to S3."""
    import zipfile, tempfile
    logger.info("Uploading code to S3...")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            for base in ["algotrading", "scripts"]:
                for dirpath, _, filenames in os.walk(os.path.join(ROOT, base)):
                    for fname in filenames:
                        if fname.endswith(".pyc"):
                            continue
                        fpath = os.path.join(dirpath, fname)
                        zf.write(fpath, os.path.relpath(fpath, ROOT))
            for f in ["pyproject.toml", "requirements.txt"]:
                zf.write(os.path.join(ROOT, f), f)
        tmp_path = tmp.name

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    s3.upload_file(tmp_path, bucket, CODE_S3_KEY)
    os.unlink(tmp_path)
    logger.info(f"Code uploaded to s3://{bucket}/{CODE_S3_KEY}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job",        default="backfill", choices=[
                            "backfill", "resample-daily",
                            "sharadar-sf1", "sharadar-daily",
                            "sharadar-tickers", "sharadar-meta", "sharadar-all",
                        ])
    parser.add_argument("--start",      default="2016-03-30")
    parser.add_argument("--end",        default="2026-03-27")
    parser.add_argument("--env",        default="algotrading/prod")
    parser.add_argument("--workers",    type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--tickers",    nargs="+", default=None)
    parser.add_argument("--year",       default=None)
    parser.add_argument("--month",      default=None)
    parser.add_argument("--start-year", default=None)
    parser.add_argument("--end-year",   default=None)
    parser.add_argument("--write-dim",  action="store_true")
    parser.add_argument("--setup",      action="store_true", help="Run first-time setup on the instance")
    parser.add_argument("--no-stop",    action="store_true", help="Keep instance running after job completes")
    parser.add_argument("--no-chunk",   action="store_true", help="SF1: download full table at once (for coverage testing)")
    args = parser.parse_args()

    bucket = os.environ["S3_BUCKET"]

    runner = EC2Runner()
    runner.start()

    if args.setup:
        runner.setup()

    # Upload current local code to S3, EC2 pulls it down
    upload_code(bucket)
    runner.run(
        f"aws s3 cp s3://{bucket}/{CODE_S3_KEY} /tmp/code.zip && "
        f"unzip -o /tmp/code.zip -d {runner.REPO_DIR} && "
        f"{runner.REPO_DIR}/.venv/bin/pip install -q -r {runner.REPO_DIR}/requirements.txt && "
        f"{runner.REPO_DIR}/.venv/bin/pip install -q {runner.REPO_DIR}"
    )

    env_vars = (
        f"S3_BUCKET='{os.environ['S3_BUCKET']}' "
        f"AWS_ACCESS_KEY_ID='{os.environ['AWS_ACCESS_KEY_ID']}' "
        f"AWS_SECRET_ACCESS_KEY='{os.environ['AWS_SECRET_ACCESS_KEY']}' "
        f"AWS_REGION='{os.environ.get('AWS_REGION', 'us-east-1')}' "
        f"NASDAQ_DATA_LINK_API_KEY='{os.environ.get('NASDAQ_DATA_LINK_API_KEY', '')}' "
    )
    base = f"cd {runner.REPO_DIR} && {env_vars} {runner.REPO_DIR}/.venv/bin/python"

    if args.job == "backfill":
        cmd = (
            f"{base} scripts/backfill.py "
            f"--start {args.start} --end {args.end} --env {args.env} "
            f"--workers {args.workers} --batch-size {args.batch_size}"
        )
        if args.write_dim:
            cmd += " --write-dim"
        if args.tickers:
            cmd += f" --tickers {' '.join(args.tickers)}"
        cmd = f"POLYGON_API_KEY='{os.environ['POLYGON_API_KEY']}' " + cmd

    elif args.job == "resample-daily":
        cmd = f"{base} scripts/resample_daily.py --env {args.env} --workers 8"
        if args.year:
            cmd += f" --year {args.year}"
        if args.month:
            cmd += f" --month {args.month}"
        if args.start_year:
            cmd += f" --start-year {args.start_year}"
        if args.end_year:
            cmd += f" --end-year {args.end_year}"

    elif args.job == "sharadar-tickers":
        cmd = f"{base} scripts/backfill_sharadar_tickers.py --env {args.env}"

    elif args.job == "sharadar-sf1":
        cmd = f"{base} scripts/backfill_sharadar_sf1.py --env {args.env} --workers {args.workers}"
        if args.no_chunk:
            cmd += " --no-chunk"

    elif args.job == "sharadar-daily":
        cmd = f"{base} scripts/backfill_sharadar_daily.py --env {args.env}"

    elif args.job == "sharadar-meta":
        cmd = f"{base} scripts/backfill_sharadar_meta.py --env {args.env}"

    elif args.job == "sharadar-all":
        cmd = (
            f"{base} scripts/backfill_sharadar_tickers.py --env {args.env} && "
            f"{base} scripts/backfill_sharadar_sf1.py --env {args.env} --workers {args.workers} && "
            f"{base} scripts/backfill_sharadar_daily.py --env {args.env} && "
            f"{base} scripts/backfill_sharadar_meta.py --env {args.env}"
        )

    runner.run(cmd, timeout=14400)  # 4h max for full sharadar download

    if not args.no_stop:
        runner.stop()


if __name__ == "__main__":
    main()
