"""
Generic signal quality diagnostics. Works for any signal panel at any frequency.

Import and call run_checks() from any compute script or notebook.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)
SEP = "=" * 60


def check_coverage(df: pd.DataFrame, signal_col: str, freq: str) -> None:
    print(f"\n{SEP}\n1. COVERAGE\n{SEP}")
    print(f"  Rows        : {len(df):,}")
    print(f"  Tickers     : {df['ticker'].nunique():,}")
    print(f"  Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Periods     : {df['date'].nunique()}  (freq={freq})")
    for col in [signal_col, "ME", "ret", "close"]:
        if col in df.columns:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<12} coverage : {pct:.1f}%")
    n_per_period = df.groupby("date")["ticker"].count()
    print(f"\n  Tickers/period  min={n_per_period.min()}  median={n_per_period.median():.0f}  max={n_per_period.max()}")


def check_outliers(df: pd.DataFrame, signal_col: str) -> None:
    print(f"\n{SEP}\n2. OUTLIERS — {signal_col} distribution\n{SEP}")
    desc = df[signal_col].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    for k, v in desc.items():
        print(f"  {k:<8}: {v:.6f}")
    for threshold in [1, 5]:
        n = (df[signal_col].abs() > threshold).sum()
        print(f"  |{signal_col}| > {threshold:<3}: {n:,} rows ({n / len(df) * 100:.2f}%)")


def check_date_gaps(df: pd.DataFrame, freq: str, max_show: int = 10) -> None:
    print(f"\n{SEP}\n3. DATE GAPS — tickers with missing periods\n{SEP}")
    period_freq = {"monthly": "M", "daily": "D", "weekly": "W"}.get(freq, "M")
    # Use actual periods present in the data (not calendar range) to avoid counting holidays/weekends
    all_periods = df["date"].dt.to_period(period_freq).unique()
    n_total = len(all_periods)
    gaps = []
    for ticker, grp in df.groupby("ticker"):
        ticker_periods = grp["date"].dt.to_period(period_freq).nunique()
        missing = n_total - ticker_periods
        if missing > 0:
            gaps.append((ticker, missing))
    gaps.sort(key=lambda x: -x[1])
    print(f"  Tickers with any gap : {len(gaps):,} / {df['ticker'].nunique():,}")
    if gaps:
        print(f"  Worst offenders (top {max_show}):")
        for ticker, n in gaps[:max_show]:
            print(f"    {ticker:<12} missing {n} periods")


def check_lookahead(df: pd.DataFrame, signal_col: str) -> None:
    print(f"\n{SEP}\n4. LOOK-AHEAD SANITY\n{SEP}")
    first_signal = df[df[signal_col].notna()].groupby("ticker")["date"].min().rename("first_date")
    merged = df.merge(first_signal, on="ticker", how="left")
    wrong = merged[(merged["date"] < merged["first_date"]) & merged[signal_col].notna()]
    status = "✓ OK" if len(wrong) == 0 else "✗ ISSUE"
    print(f"  {signal_col} before first valid filing : {len(wrong):,}  {status}")


def check_me_sanity(df: pd.DataFrame) -> None:
    if "ME" not in df.columns:
        return
    print(f"\n{SEP}\n5. MARKET EQUITY SANITY\n{SEP}")
    neg = (df["ME"] <= 0).sum()
    print(f"  ME <= 0  : {neg:,} rows")
    print(f"  ME p1    : ${df['ME'].quantile(0.01) / 1e6:.1f}M")
    print(f"  ME median: ${df['ME'].median() / 1e9:.2f}B")
    print(f"  ME p99   : ${df['ME'].quantile(0.99) / 1e9:.1f}B")


def check_return_sanity(df: pd.DataFrame) -> None:
    if "ret" not in df.columns:
        return
    print(f"\n{SEP}\n6. RETURN SANITY\n{SEP}")
    valid = df["ret"].dropna()
    extreme = (valid.abs() > 1).sum()
    print(f"  |ret| > 100% : {extreme:,} rows ({extreme / len(valid) * 100:.2f}%)")
    print(f"  ret min      : {valid.min():.4f}")
    print(f"  ret max      : {valid.max():.4f}")
    print(f"  ret mean     : {valid.mean():.4f}")
    print(f"  ret std      : {valid.std():.4f}")


def run_checks(df: pd.DataFrame, signal_col: str = "CCH", freq: str = "monthly") -> None:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    check_coverage(df, signal_col, freq)
    check_outliers(df, signal_col)
    check_date_gaps(df, freq)
    check_lookahead(df, signal_col)
    check_me_sanity(df)
    check_return_sanity(df)
    print(f"\n{SEP}\nDone.\n{SEP}")
