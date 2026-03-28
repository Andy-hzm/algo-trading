import os
import time
import logging
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

import pandas as pd
import urllib3
from polygon import RESTClient
from polygon.exceptions import BadResponse

logger = logging.getLogger(__name__)

DateLike = Union[str, date, datetime]


class PolygonClient:
    """
    Wrapper around the Polygon REST API for fetching OHLCV price bars.

    Parallelism design:
        - get_bars_bulk() uses ThreadPoolExecutor for I/O overlap within a process
        - For multi-core parallelism, call get_bars_bulk() from multiple processes
          (see scripts/backfill.py) — each process instantiates its own PolygonClient
    """

    def __init__(self, api_key: str = None, max_workers: int = 10):
        """
        Args:
            api_key:     Polygon API key. Falls back to POLYGON_API_KEY env var.
            max_workers: Max threads for get_bars_bulk (default 10).
        """
        api_key = api_key or os.environ.get("POLYGON_API_KEY")
        if not api_key:
            raise ValueError("POLYGON_API_KEY not set")
        self._client = RESTClient(api_key=api_key)
        # Increase connection pool to match thread count — prevents
        # "Connection pool is full" warnings under heavy threading
        self._client.client = urllib3.PoolManager(
            num_pools=max_workers,
            maxsize=max_workers,
            headers=self._client.headers,
        )
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bars(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        timespan: str = "hour",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single ticker.

        Returns a DataFrame indexed by UTC timestamp with columns:
            ticker, open, high, low, close, volume, vwap, transactions

        Returns empty DataFrame if no data is available.
        """
        bars = self._fetch_with_retry(ticker, start, end, timespan, multiplier, adjusted)
        if not bars:
            logger.warning(f"No bars returned for {ticker} ({start} -> {end})")
            return pd.DataFrame()
        return self._to_dataframe(ticker, bars)

    def get_bars_bulk(
        self,
        tickers: List[str],
        start: DateLike,
        end: DateLike,
        timespan: str = "hour",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch bars for multiple tickers using threads (I/O overlap).

        Uses ThreadPoolExecutor with max_workers threads — good for overlapping
        network wait time within a single process. For multi-core parallelism,
        call this from multiple processes (see scripts/backfill.py).

        Returns a single combined DataFrame sorted by timestamp.
        """
        frames = []
        failed = []

        def _fetch(ticker):
            return ticker, self.get_bars(ticker, start, end, timespan, multiplier, adjusted)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_fetch, t): t for t in tickers}
            for i, future in enumerate(as_completed(futures)):
                ticker = futures[future]
                try:
                    _, df = future.result()
                    if not df.empty:
                        frames.append(df)
                    logger.info(f"[{i+1}/{len(tickers)}] {ticker} — {len(df)} bars")
                except Exception as e:
                    logger.error(f"{ticker} failed: {e}")
                    failed.append(ticker)

        if failed:
            logger.warning(f"{len(failed)} tickers failed: {failed}")
        if not frames:
            return pd.DataFrame()

        return pd.concat(frames).sort_index()

    def get_tickers(
        self,
        exchanges: List[str] = None,
        market: str = "stocks",
        type: str = "CS",
        active: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch the ticker universe and return as a clean DataFrame.

        Args:
            exchanges: List of MIC exchange codes e.g. ['XNYS', 'XNAS'].
                       Defaults to NYSE + Nasdaq.
            market:    'stocks' | 'crypto' | 'fx'
            type:      'CS' (common stock) | 'ETF' | 'ADRC' etc.
            active:    Only return currently active tickers.

        Returns a DataFrame with columns:
            ticker, name, primary_exchange, type, active, cik, currency_name
        """
        if exchanges is None:
            exchanges = ["XNYS", "XNAS"]

        raw = []
        for exchange in exchanges:
            for t in self._client.list_tickers(
                market=market,
                exchange=exchange,
                type=type,
                active=active,
                limit=1000,
            ):
                raw.append(vars(t))

        if not raw:
            return pd.DataFrame()

        df = pd.DataFrame(raw)
        keep = ["ticker", "name", "primary_exchange", "type", "active", "cik", "currency_name"]
        keep = [c for c in keep if c in df.columns]
        return df[keep].drop_duplicates(subset="ticker").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(
        self,
        ticker: str,
        start: DateLike,
        end: DateLike,
        timespan: str,
        multiplier: int,
        adjusted: bool,
        retries: int = 3,
    ) -> list:
        for attempt in range(retries):
            try:
                return list(
                    self._client.list_aggs(
                        ticker=ticker,
                        multiplier=multiplier,
                        timespan=timespan,
                        from_=self._to_str(start),
                        to=self._to_str(end),
                        adjusted=adjusted,
                        sort="asc",
                        limit=50000,
                    )
                )
            except BadResponse as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"{ticker} attempt {attempt+1} failed: {e}. Retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"{ticker} failed after {retries} attempts: {e}")
                    return []

    def _to_dataframe(self, ticker: str, bars: list) -> pd.DataFrame:
        df = pd.DataFrame([vars(b) for b in bars])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df.insert(0, "ticker", ticker)
        return df[["ticker", "open", "high", "low", "close", "volume", "vwap", "transactions"]]

    @staticmethod
    def _to_str(d: DateLike) -> str:
        if isinstance(d, (date, datetime)):
            return d.strftime("%Y-%m-%d")
        return d
