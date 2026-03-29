"""
Run backfill on EC2 via SSM. No SSH needed.

Usage:
    python scripts/run_ec2.py                          # full backfill, prod
    python scripts/run_ec2.py --setup                  # first-time setup
    python scripts/run_ec2.py --env dev --tickers AAPL MSFT
    python scripts/run_ec2.py --no-stop                # keep instance running after job
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algotrading.infra.ec2_runner import EC2Runner

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2016-03-30")
    parser.add_argument("--end",        default="2026-03-27")
    parser.add_argument("--env",        default="prod", choices=["prod", "dev"])
    parser.add_argument("--workers",    type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--tickers",    nargs="+", default=None)
    parser.add_argument("--write-dim",  action="store_true")
    parser.add_argument("--setup",      action="store_true", help="Run first-time setup on the instance")
    parser.add_argument("--no-stop",    action="store_true", help="Keep instance running after job completes")
    args = parser.parse_args()

    runner = EC2Runner()
    runner.start()

    if args.setup:
        runner.setup()

    # Build the backfill command — identical to running it locally
    cmd = (
        f"cd {runner.REPO_DIR} && git pull && "
        f"{runner.REPO_DIR}/.venv/bin/pip install -q {runner.REPO_DIR} && "
        f"POLYGON_API_KEY={os.environ['POLYGON_API_KEY']} "
        f"S3_BUCKET={os.environ['S3_BUCKET']} "
        f"AWS_ACCESS_KEY_ID={os.environ['AWS_ACCESS_KEY_ID']} "
        f"AWS_SECRET_ACCESS_KEY={os.environ['AWS_SECRET_ACCESS_KEY']} "
        f"AWS_REGION={os.environ.get('AWS_REGION', 'us-east-1')} "
        f"{runner.REPO_DIR}/.venv/bin/python scripts/backfill.py "
        f"--start {args.start} --end {args.end} --env {args.env} "
        f"--workers {args.workers} --batch-size {args.batch_size}"
    )
    if args.write_dim:
        cmd += " --write-dim"
    if args.tickers:
        cmd += f" --tickers {' '.join(args.tickers)}"

    runner.run(cmd)

    if not args.no_stop:
        runner.stop()


if __name__ == "__main__":
    main()
