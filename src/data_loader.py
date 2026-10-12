"""
Data loading from Massive REST API with local Parquet caching.
Fetches unadjusted and adjusted daily OHLCV for U.S. common stocks.
Universe = top UNIVERSE_SIZE stocks by ADDV among CS tickers listed on major exchanges.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import (
    ADDV_DAYS,
    BACKTEST_DAYS,
    BACKTEST_END_DATE,
    FETCH_START_DATE,
    INVESTMENT_LEVEL,
    REBAL_PERIOD,
    UNIVERSE_SIZE,
)

load_dotenv()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_PATH = DATA_DIR / "panel_unadj.parquet"
PANEL_ADJ_PATH = DATA_DIR / "panel_adj.parquet"
TICKERS_PATH = DATA_DIR / "tickers.parquet"


def _get_client():
    """Instantiate Massive REST client from environment."""
    from massive import RESTClient  # type: ignore

    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MASSIVE_API_KEY not set. Copy .env.example to .env and fill in your key."
        )
    return RESTClient(api_key=api_key)


def fetch_universe_tickers(
    min_tickers: int = 500, cache: bool = True
) -> list[str]:
    """
    Retrieve U.S. common stock tickers from Massive.
    Filters to type='CS' (common stock / class shares), major exchanges only.
    Returns a list of ticker strings suitable for bar requests.
    """
    if cache and TICKERS_PATH.exists():
        df = pd.read_parquet(TICKERS_PATH, engine="fastparquet")
        tickers = df["ticker"].tolist()
        logger.info("Loaded %d tickers from cache.", len(tickers))
        return tickers

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    records = []
    logger.info("Fetching CS tickers from Massive …")
    # MIC exchange codes for NYSE (XNYS), NASDAQ (XNAS), NYSE American/AMEX (XASE)
    # active=True gives currently listed stocks; historical delisted stocks are excluded
    # (survivorship bias — noted in RESULTS.md Deviations)
    for t in client.list_tickers(
        market="stocks",
        type="CS",
        active=True,
        limit=1000,
    ):
        exch = getattr(t, "primary_exchange", None)
        if exch in ("XNYS", "XNAS", "XASE"):
            records.append({"ticker": t.ticker, "name": getattr(t, "name", ""), "type": t.type})

    if not records:
        raise RuntimeError(
            "list_tickers returned 0 results – check API key and market/type filters."
        )

    df = pd.DataFrame(records).drop_duplicates("ticker").reset_index(drop=True)
    logger.info(
        "Fetched %d tickers. Fields: %s. Sample: %s",
        len(df),
        df.columns.tolist(),
        df["ticker"].head(5).tolist(),
    )

    if cache:
        df.to_parquet(TICKERS_PATH, index=False, engine="fastparquet")

    return df["ticker"].tolist()


def _fetch_aggs_for_ticker(
    client,
    ticker: str,
    from_date: str,
    to_date: str,
    adjusted: bool,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch daily bars for one ticker; returns DataFrame with date index."""
    for attempt in range(max_retries):
        try:
            rows = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=from_date,
                to=to_date,
                adjusted=adjusted,
                limit=50000,
            ):
                rows.append(a)
            if not rows:
                return pd.DataFrame()
            data = {
                "date": pd.to_datetime([r.timestamp for r in rows], unit="ms", utc=True)
                .tz_convert("America/New_York")
                .normalize()
                .date,
                "open": [r.open for r in rows],
                "close": [r.close for r in rows],
                "volume": [r.volume for r in rows],
            }
            df = pd.DataFrame(data).drop_duplicates("date").set_index("date")
            df.index = pd.DatetimeIndex(df.index)
            return df.sort_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Attempt %d/%d for %s (%s): %s", attempt + 1, max_retries, ticker, "adj" if adjusted else "unadj", exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return pd.DataFrame()


def download_all_bars(
    tickers: list[str],
    from_date: str = FETCH_START_DATE,
    to_date: str = BACKTEST_END_DATE,
    cache: bool = True,
    max_tickers: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download unadjusted and adjusted daily bars for all tickers.
    Returns two DataFrames each with MultiIndex (date, ticker) and columns [open, close, volume].
    """
    if cache and PANEL_PATH.exists() and PANEL_ADJ_PATH.exists():
        panel_unadj = pd.read_parquet(PANEL_PATH, engine="fastparquet")
        panel_adj = pd.read_parquet(PANEL_ADJ_PATH, engine="fastparquet")
        logger.info(
            "Loaded cached panels: %d unadj rows, %d adj rows, tickers: %d",
            len(panel_unadj),
            len(panel_adj),
            panel_unadj.index.get_level_values("ticker").nunique(),
        )
        return panel_unadj, panel_adj

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if max_tickers:
        tickers = tickers[:max_tickers]

    client = _get_client()

    # Smoke test: verify one ticker before bulk download
    _smoke_test_ticker = tickers[0]
    sample = _fetch_aggs_for_ticker(client, _smoke_test_ticker, from_date, to_date, adjusted=False)
    if sample.empty:
        raise RuntimeError(
            f"Smoke test failed: no data for {_smoke_test_ticker}. Check date range and API key entitlements."
        )
    logger.info(
        "Smoke test OK: ticker=%s rows=%d fields=%s date_range=[%s, %s] sample_close=%.2f",
        _smoke_test_ticker,
        len(sample),
        sample.columns.tolist(),
        sample.index[0].date(),
        sample.index[-1].date(),
        float(sample["close"].iloc[-1]),
    )

    t0 = time.perf_counter()
    unadj_frames, adj_frames = [], []

    for i, ticker in enumerate(tickers):
        df_u = _fetch_aggs_for_ticker(client, ticker, from_date, to_date, adjusted=False)
        df_a = _fetch_aggs_for_ticker(client, ticker, from_date, to_date, adjusted=True)
        if df_u.empty or df_a.empty:
            continue
        df_u["ticker"] = ticker
        df_a["ticker"] = ticker
        unadj_frames.append(df_u)
        adj_frames.append(df_a)
        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            remaining = (len(tickers) - i - 1) / rate / 60
            logger.info(
                "Downloaded %d/%d tickers (%.1f/min, ~%.0f min remaining).",
                i + 1,
                len(tickers),
                rate * 60,
                remaining,
            )

    if not unadj_frames:
        raise RuntimeError("No data downloaded – check tickers and API entitlements.")

    panel_unadj = (
        pd.concat(unadj_frames)
        .reset_index()
        .rename(columns={"index": "date"})
        .set_index(["date", "ticker"])
        .sort_index()
    )
    panel_adj = (
        pd.concat(adj_frames)
        .reset_index()
        .rename(columns={"index": "date"})
        .set_index(["date", "ticker"])
        .sort_index()
    )

    n_tickers = panel_unadj.index.get_level_values("ticker").nunique()
    date_range = panel_unadj.index.get_level_values("date")
    logger.info(
        "Download complete: %d tickers, %d rows, dates [%s, %s].",
        n_tickers,
        len(panel_unadj),
        date_range.min(),
        date_range.max(),
    )

    if cache:
        panel_unadj.to_parquet(PANEL_PATH, engine="fastparquet")
        panel_adj.to_parquet(PANEL_ADJ_PATH, engine="fastparquet")

    return panel_unadj, panel_adj


def build_price_arrays(
    panel_unadj: pd.DataFrame,
    panel_adj: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Pivot panel DataFrames into date × ticker matrices for:
      - unadj_open (P^O), unadj_close (P^C), unadj_volume (V)
      - adj_open (P^AO), adj_close (P^AC)
    Missing values left as NaN (no imputation per paper).
    """
    unadj_open = panel_unadj["open"].unstack("ticker")
    unadj_close = panel_unadj["close"].unstack("ticker")
    unadj_vol = panel_unadj["volume"].unstack("ticker")
    adj_open = panel_adj["open"].unstack("ticker")
    adj_close = panel_adj["close"].unstack("ticker")

    # Align all to common date × ticker grid
    tickers = unadj_close.columns.intersection(adj_close.columns)
    dates = unadj_close.index.intersection(adj_close.index)

    arrays = {
        "unadj_open": unadj_open.loc[dates, tickers],
        "unadj_close": unadj_close.loc[dates, tickers],
        "unadj_vol": unadj_vol.loc[dates, tickers],
        "adj_open": adj_open.loc[dates, tickers],
        "adj_close": adj_close.loc[dates, tickers],
    }

    logger.info(
        "Price arrays built: %d dates × %d tickers.",
        len(dates),
        len(tickers),
    )
    return arrays


def compute_addv(
    unadj_close: pd.DataFrame,
    unadj_vol: pd.DataFrame,
    window: int = ADDV_DAYS,
) -> pd.DataFrame:
    """
    Compute rolling ADDV = (1/d) * sum_{r=1}^d V_{i,s+r} * P^C_{i,s+r}
    using only prior data (look-ahead free).
    Returns DataFrame same shape as unadj_close.
    Section 3.4.2: ADDV uses the immediately preceding d=21 trading days.
    """
    dv = unadj_vol * unadj_close
    # rolling(window).mean() uses rows [t-window, t-1] → value available at end of day t-1
    # shift(1) makes it available at start of day t (prior data only)
    addv = dv.rolling(window=window, min_periods=window).mean().shift(1)
    return addv


def get_trading_dates(
    price_arrays: dict[str, pd.DataFrame],
    end_date: str = BACKTEST_END_DATE,
    n_days: int = BACKTEST_DAYS,
) -> pd.DatetimeIndex:
    """
    Return the last n_days trading dates in the price arrays up to end_date.
    Section 3.4.2: 1260 trading days ending 2014-09-05 inclusive.
    """
    all_dates = pd.DatetimeIndex(price_arrays["unadj_close"].index)
    end = pd.Timestamp(end_date)
    dates = all_dates[all_dates <= end]
    if len(dates) < n_days:
        raise ValueError(
            f"Only {len(dates)} trading days available up to {end_date}; need {n_days}."
        )
    return dates[-n_days:]
