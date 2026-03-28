"""
Backfill script — fetches full price history for the ticker universe and writes to S3.

Usage:
    cd algo-trading
    python scripts/backfill.py                          # full universe, 10yr, prod
    python scripts/backfill.py --env dev               # write to dev/
    python scripts/backfill.py --start 2016-03-30      # custom start date
    python scripts/backfill.py --tickers AAPL MSFT     # specific tickers only
    python scripts/backfill.py --workers 5             # number of processes (default 5)
    python scripts/backfill.py --batch-size 50         # tickers per batch (default 50)
    python scripts/backfill.py --write-dim             # refresh dim table before backfill
"""

import argparse
import logging
import os
import sys
import time
from multiprocessing import Pool

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from algotrading.db.polygon_client import PolygonClient
from algotrading.db.s3_client import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)


# -- worker: fetches a batch of tickers, writes all at once ---------------

def _fetch_batch(args):
    batch, start, end, env = args
    from algotrading.db.polygon_client import PolygonClient
    from algotrading.db.s3_client import S3Client

    pc = PolygonClient()          # threaded inside: 10 concurrent requests
    s3 = S3Client(prefix=env)

    df = pc.get_bars_bulk(batch, start, end)   # fetch all tickers in batch
    if not df.empty:
        s3.write_bars_batch(df)                # one S3 PUT per (ticker, month)
        logger.info(f"Batch of {len(batch)} tickers — {len(df)} rows written")
    else:
        logger.warning(f"Batch returned no data: {batch}")


# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2016-03-30")
    parser.add_argument("--end",        default="2026-03-27")
    parser.add_argument("--tickers",    nargs="+", default=None)
    parser.add_argument("--workers",    type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--env",        default="prod", choices=["prod", "dev"])
    parser.add_argument("--write-dim",  action="store_true")
    args = parser.parse_args()

    pc = PolygonClient()
    s3 = S3Client(prefix=args.env)

    if args.write_dim:
        logger.info("Fetching dim table...")
        dim = pc.get_tickers()
        s3.write_dim(dim)
        logger.info(f"Dim table written ({len(dim)} tickers)")

    if args.tickers:
        tickers = args.tickers
    else:
        dim = s3.read_dim()
        if dim.empty:
            logger.error("Dim table not found in S3. Use --write-dim to create it.")
            sys.exit(1)
        tickers = dim["ticker"].tolist()

    # Split tickers into batches
    batches = [tickers[i:i + args.batch_size] for i in range(0, len(tickers), args.batch_size)]
    tasks = [(b, args.start, args.end, args.env) for b in batches]

    logger.info(
        f"Backfilling {len(tickers)} tickers in {len(batches)} batches "
        f"| {args.start} → {args.end} | {args.workers} workers | env={args.env}"
    )

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        pool.map(_fetch_batch, tasks)
    elapsed = time.time() - t0

    logger.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
