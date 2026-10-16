from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"          # tracked by git
FEATURES_OUT = DATA / "features"    # git-ignored

OHLCV = DATA / "prices" / "ohlcv_daily.csv"
OHLCV_WARMUP = DATA / "prices" / "ohlcv_warmup.csv"
MACRO_FULL = DATA / "macro" / "fred_daily_full.csv"
NEWS = DATA / "news" / "news_final30.parquet"
NEWS_COVERAGE = DATA / "news" / "news_daily_coverage.csv"
TICKERS_CSV = DATA / "universe" / "final_30_tickers_v2.csv"

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = None   # fill with the commit hash after first download
FINBERT_MAX_LEN = 160

LSTM_LOOKBACK = 20
SENT_WINDOW = 60
SENT_MIN_PERIODS = 20   # pre-FIRST_MODEL_DATE days exist as LSTM inputs only

def trading_dates():
    d = pd.read_csv(OHLCV, usecols=["date"], parse_dates=["date"])["date"]
    return pd.DatetimeIndex(sorted(d.unique()))

_TD = trading_dates()
FIRST_MODEL_DATE = _TD[_TD.year == 2019][60]            # 61st trading day
FIRST_LSTM_INPUT_DATE = _TD[_TD.get_loc(FIRST_MODEL_DATE) - (LSTM_LOOKBACK - 1)]