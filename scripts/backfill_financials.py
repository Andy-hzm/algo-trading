"""
Backfill fundamental financials for the ticker universe.

Usage:
    python scripts/backfill_financials.py
    python scripts/backfill_financials.py --env algotrading/dev
    python scripts/backfill_financials.py --start 2016-01-01 --end 2026-01-01
    python scripts/backfill_financials.py --tickers AAPL MSFT
    python scripts/backfill_financials.py --timeframe annual
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

from algotrading.infra.polygon_client import PolygonClient
from algotrading.infra.s3_client import S3Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _fetch_batch(args):
    batch, start, end, timeframe, env = args
    from algotrading.infra.polygon_client import PolygonClient
    from algotrading.infra.s3_client import S3Client

    pc = PolygonClient()
    s3 = S3Client(prefix=env)

    df = pc.get_financials_bulk(batch, start, end, timeframe)
    if not df.empty:
        s3.write_financials_batch(df)
        logger.info(f"Batch of {len(batch)} tickers — {len(df)} filings written")
    else:
        logger.warning(f"Batch returned no data: {batch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",      default="2016-01-01")
    parser.add_argument("--end",        default="2026-01-01")
    parser.add_argument("--tickers",    nargs="+", default=None)
    parser.add_argument("--timeframe",  default="quarterly", choices=["quarterly", "annual"])
    parser.add_argument("--workers",    type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--env",        default="algotrading/prod", choices=["algotrading/prod", "algotrading/dev"])
    args = parser.parse_args()

    s3 = S3Client(prefix=args.env)

    if args.tickers:
        tickers = args.tickers
    else:
        dim = s3.read_dim()
        if dim.empty:
            logger.error("Dim table not found. Run backfill.py --write-dim first.")
            sys.exit(1)
        tickers = dim["ticker"].tolist()

    batches = [tickers[i:i + args.batch_size] for i in range(0, len(tickers), args.batch_size)]
    tasks = [(b, args.start, args.end, args.timeframe, args.env) for b in batches]

    logger.info(
        f"Backfilling financials for {len(tickers)} tickers in {len(batches)} batches "
        f"| {args.start} → {args.end} | timeframe={args.timeframe} | {args.workers} workers | env={args.env}"
    )

    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        pool.map(_fetch_batch, tasks)
    elapsed = time.time() - t0

    logger.info(f"Done in {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
