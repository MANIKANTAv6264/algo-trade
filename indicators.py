"""
indicators.py
All technical indicators used as ML features.
IMPORTANT: Always call on SPLIT data only — never on full dataset.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def _ema(s, span):  return s.ewm(span=span, adjust=False).mean()
def _sma(s, period): return s.rolling(window=period).mean()


def calculate_rsi(df, period=14):
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period-1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(df, fast=12, slow=26, signal=9):
    macd = _ema(df["close"], fast) - _ema(df["close"], slow)
    sig  = _ema(macd, signal)
    return macd, sig, macd - sig


def calculate_bollinger(df, period=20, std=2):
    mid   = _sma(df["close"], period)
    sigma = df["close"].rolling(period).std()
    return mid + std*sigma, mid, mid - std*sigma


def calculate_atr(df, period=14):
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_adx(df, period=14):
    plus_dm  = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    atr      = calculate_atr(df, period)
    eps      = 1e-10
    plus_di  = 100 * plus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + eps)
    minus_di = 100 * minus_dm.ewm(alpha=1/period, adjust=False).mean() / (atr + eps)
    dx       = (abs(plus_di - minus_di) / (plus_di + minus_di + eps)) * 100
    adx      = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx, plus_di, minus_di


def calculate_stochastic(df, k=14, d=3):
    low_min  = df["low"].rolling(k).min()
    high_max = df["high"].rolling(k).max()
    denom    = (high_max - low_min).replace(0, np.nan)
    K        = 100 * (df["close"] - low_min) / denom
    D        = K.rolling(d).mean()
    return K, D


def detect_regime(df) -> str:
    """
    Returns: 'trending' | 'ranging' | 'volatile'
    Used to adjust confidence thresholds.
    """
    if len(df) < 30:
        return 'ranging'
    last    = df.tail(20)
    adx_val = last["adx"].iloc[-1] if "adx" in last.columns else 20
    atr_val = (last["atr"] / last["close"]).iloc[-1] if "atr" in last.columns else 0.01
    avg_atr = (df["atr"] / df["close"]).rolling(60).mean().iloc[-1] if "atr" in df.columns else 0.01
    if atr_val > avg_atr * 1.5:
        return 'volatile'
    if adx_val > 25:
        return 'trending'
    return 'ranging'


# Features used for ML — selected for low multicollinearity
FEATURE_COLUMNS = [
    "rsi", "macd", "macd_signal",
    "bb_position", "bb_width",
    "stoch_k", "stoch_d",
    "adx", "plus_di", "minus_di",
    "volume_ratio", "atr_pct",
    "above_ema50", "above_ema200", "ema9_above_21", "macd_bullish",
    "price_change", "price_change_3d", "price_change_5d",
    "hl_ratio",
]


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all indicators to dataframe.
    Call ONLY on already-split data to avoid leakage.
    """
    df = df.copy()

    # EMA — only 9, 21, 50, 200 (removed sma20/sma50 — correlated)
    df["ema9"]   = _ema(df["close"], 9)
    df["ema21"]  = _ema(df["close"], 21)
    df["ema50"]  = _ema(df["close"], 50)
    df["ema200"] = _ema(df["close"], 200)

    # RSI
    df["rsi"] = calculate_rsi(df)

    # MACD (removed macd_hist — redundant with macd-signal)
    df["macd"], df["macd_signal"], _ = calculate_macd(df)

    # Bollinger
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = calculate_bollinger(df)
    bb_range          = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    df["bb_width"]    = bb_range / df["bb_mid"].replace(0, np.nan)
    df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range

    # ATR as % of price (normalised — avoids price-scale issues)
    df["atr"]     = calculate_atr(df)
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)

    # ADX
    df["adx"], df["plus_di"], df["minus_di"] = calculate_adx(df)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = calculate_stochastic(df)

    # Volume ratio
    avg_vol           = df["volume"].rolling(20).mean()
    df["volume_ratio"]= df["volume"] / avg_vol.replace(0, np.nan)

    # Price momentum
    df["price_change"]    = df["close"].pct_change()
    df["price_change_3d"] = df["close"].pct_change(3)
    df["price_change_5d"] = df["close"].pct_change(5)

    # Boolean flags (as int)
    df["above_ema50"]   = (df["close"] > df["ema50"]).astype(int)
    df["above_ema200"]  = (df["close"] > df["ema200"]).astype(int)
    df["ema9_above_21"] = (df["ema9"]  > df["ema21"]).astype(int)
    df["macd_bullish"]  = (df["macd"]  > df["macd_signal"]).astype(int)

    # Candle body ratio
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def get_support_resistance(df, lookback=20):
    recent = df.tail(lookback)
    return float(recent["low"].min()), float(recent["high"].max())


def get_feature_importance(model, feature_names):
    """Extract feature importance from trained model."""
    try:
        # VotingClassifier — get from RF estimator
        for name, est in model.named_estimators_.items():
            if hasattr(est, 'feature_importances_'):
                imp = est.feature_importances_
                return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)
    except Exception:
        pass
    return []
