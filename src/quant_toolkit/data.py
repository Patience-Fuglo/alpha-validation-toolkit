"""Real daily OHLCV loading for NVDA/MSFT-style single-name equity research.

Yahoo Finance (via yfinance) is the primary source; it free-tier rate-limits
aggressively (HTTP 429) under repeated calls, so every pull is cached to disk
and retried with backoff before falling back to Stooq's free daily-bar CSV
endpoint. Callers never need to know which source ultimately served the data.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data"

_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class DataUnavailableError(RuntimeError):
    """Raised when no configured data source can return the requested series."""


def load_ohlcv(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_retries: int = 3,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV bars for ``ticker`` between ``start`` and ``end`` (inclusive dates, "YYYY-MM-DD").

    Returns a DataFrame indexed by tz-naive ``pd.DatetimeIndex`` (ascending),
    columns ``open, high, low, close, volume`` (all float except volume:int).

    Raises ``DataUnavailableError`` if neither Yahoo nor Stooq can serve the range.
    """
    cache_path = cache_dir / f"{ticker.upper()}_{start}_{end}.csv"
    if not force_refresh and cache_path.exists():
        return _read_cache(cache_path)

    df = _fetch_yfinance(ticker, start, end, max_retries)
    source = "yfinance"
    if df is None or df.empty:
        df = _fetch_stooq(ticker, start, end)
        source = "stooq"
    if df is None or df.empty:
        raise DataUnavailableError(
            f"Could not load {ticker} [{start}, {end}] from yfinance or Stooq."
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    print(f"[data] {ticker}: {len(df)} rows from {source}, cached to {cache_path.name}")
    return df


def _read_cache(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df[_OHLCV_COLUMNS]


def _fetch_yfinance(ticker: str, start: str, end: str, max_retries: int) -> pd.DataFrame | None:
    import yfinance as yf

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
        except Exception:
            raw = None

        if raw is not None and not raw.empty:
            return _normalize_yfinance(raw)

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s backoff on rate limits / transient errors

    return None


def _normalize_yfinance(raw: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.droplevel(1, axis=1)
    raw = raw.rename(columns=str.lower)
    df = raw[_OHLCV_COLUMNS].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def _fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    from io import StringIO

    raw = pd.read_csv(StringIO(resp.text))
    if raw.empty or "Date" not in raw.columns:
        return None

    raw["Date"] = pd.to_datetime(raw["Date"])
    raw = raw.set_index("Date").rename(columns=str.lower)
    raw.index.name = "date"
    df = raw.loc[start:end, _OHLCV_COLUMNS].copy()
    return df.sort_index()
