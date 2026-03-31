"""
Integration tests — reads real data from S3 prod.

Requires valid AWS credentials and S3_BUCKET env var.
Run with: pytest tests/test_s3_integration.py -v
"""
import os
import pytest
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from algotrading.infra.s3_client import S3Client

START = "2016-04-01"
END   = "2016-12-31"
SPOT_CHECK_TICKERS = ["AAPL", "MSFT", "JPM"]

# Expected trading hours per month (rough lower bound)
MIN_BARS_PER_MONTH = 100  # ~7hrs/day * 15 trading days


@pytest.fixture(scope="module")
def s3():
    return S3Client(prefix="prod")


# ------------------------------------------------------------------
# Dim table
# ------------------------------------------------------------------

def test_dim_table_exists(s3):
    dim = s3.read_dim()
    assert not dim.empty, "Dim table not found in S3"
    assert "ticker" in dim.columns
    assert len(dim) > 1000, f"Expected 1000+ tickers, got {len(dim)}"


# ------------------------------------------------------------------
# Spot check known tickers
# ------------------------------------------------------------------

@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_ticker_has_data(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    assert not df.empty, f"No data found for {ticker}"


@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_ticker_row_count(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    months = 9  # Apr 2016 – Dec 2016
    assert len(df) >= MIN_BARS_PER_MONTH * months, (
        f"{ticker}: only {len(df)} bars for {months} months"
    )


@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_ticker_date_range(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    assert df.index.min() >= pd.Timestamp(START, tz="UTC")
    assert df.index.max() <= pd.Timestamp(END, tz="UTC")


@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_ticker_no_nulls(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    null_counts = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    assert null_counts.sum() == 0, f"{ticker} has nulls:\n{null_counts}"


@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_ticker_ohlc_sanity(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    assert (df["high"] >= df["low"]).all(),  f"{ticker}: high < low found"
    assert (df["high"] >= df["open"]).all(), f"{ticker}: high < open found"
    assert (df["high"] >= df["close"]).all(),f"{ticker}: high < close found"
    assert (df["volume"] >= 0).all(),        f"{ticker}: negative volume found"


# ------------------------------------------------------------------
# Coverage check — no full month gaps
# ------------------------------------------------------------------

@pytest.mark.parametrize("ticker", SPOT_CHECK_TICKERS)
def test_no_missing_months(s3, ticker):
    df = s3.read_ticker(ticker, START, END)
    months_present = set(zip(df.index.year, df.index.month))
    expected = {
        (y, m)
        for y, m in pd.date_range(START, END, freq="MS").map(lambda d: (d.year, d.month))
    }
    missing = expected - months_present
    assert not missing, f"{ticker} missing months: {missing}"
