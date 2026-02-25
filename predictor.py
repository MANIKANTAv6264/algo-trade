"""
predictor.py
Core prediction engine — no leakage, calibrated confidence.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from data_fetcher import fetch_historical_data, fetch_intraday_data, fetch_live_price
from indicators   import add_all_indicators, get_support_resistance, detect_regime, get_feature_importance, FEATURE_COLUMNS
from ml_model     import train_walk_forward, predict_live, load
from config       import (INTRADAY_SL_PERCENT, INTRADAY_TARGET_PERCENT,
                           SWING_SL_PERCENT, SWING_TARGET_PERCENT, MIN_CONFIDENCE)

REGIME_THRESHOLD = {'trending': 60, 'ranging': 72, 'volatile': 80}


def calculate_trade_levels(df_ind, price, signal, mode):
    """Calculate entry, target, stop loss using ATR + support/resistance."""
    support, resistance = get_support_resistance(df_ind)

    if mode == "intraday":
        sl_p, tgt_p = INTRADAY_SL_PERCENT/100, INTRADAY_TARGET_PERCENT/100
    else:
        sl_p, tgt_p = SWING_SL_PERCENT/100, SWING_TARGET_PERCENT/100

    if signal == "BUY":
        stop_loss = round(max(price*(1-sl_p), support*0.995), 2)
        target    = round(price*(1+tgt_p), 2)
        if resistance > price:
            target = round(min(target, resistance*1.005), 2)
        buy_at = round(price, 2)
    else:
        stop_loss = round(min(price*(1+sl_p), resistance*1.005), 2)
        target    = round(price*(1-tgt_p), 2)
        buy_at    = round(price, 2)

    risk   = abs(price - stop_loss)
    reward = abs(target - price)
    rr     = round(reward/risk, 2) if risk > 0 else 0
    return buy_at, target, stop_loss, rr


def analyze_stock(symbol: str, mode="swing") -> dict:
    """Full analysis pipeline for one stock."""
    try:
        # Fetch data
        if mode == "intraday":
            df_train = fetch_historical_data(symbol, period="3mo", interval="15m")
            df_live  = fetch_intraday_data(symbol, "5m") or df_train
        else:
            df_train = fetch_historical_data(symbol, period="2y",  interval="1d")
            df_live  = fetch_historical_data(symbol, period="6mo", interval="1d")

        if df_train is None or len(df_train) < 150:
            return None

        # Load or train model
        model, scaler = load(symbol, mode)
        if model is None:
            model, scaler, fold_metrics, _ = train_walk_forward(df_train, symbol, mode)
            if model is None:
                return None

        # Regime detection — adjust threshold
        df_ind = add_all_indicators(df_live.tail(60).copy())
        regime = detect_regime(df_ind)
        threshold = REGIME_THRESHOLD.get(regime, MIN_CONFIDENCE)

        # Predict
        signal, confidence = predict_live(df_live, model, scaler)

        if signal == "HOLD" or confidence < threshold:
            return None

        # Live price
        price = fetch_live_price(symbol) or float(df_live["close"].iloc[-1])

        # Trade levels
        buy_at, target, stop_loss, rr = calculate_trade_levels(df_ind, price, signal, mode)

        # Hold duration for swing
        hold = ""
        if mode == "swing":
            hold = "1-2 weeks" if confidence > 75 else "2-4 weeks"

        label = ("🟢 STRONG"   if confidence >= 80
                 else "🟡 MODERATE" if confidence >= 65
                 else "🔴 WEAK")

        return {
            "symbol":          symbol,
            "mode":            mode,
            "signal":          signal,
            "current_price":   round(price, 2),
            "buy_at":          buy_at,
            "target":          target,
            "stop_loss":       stop_loss,
            "rr_ratio":        rr,
            "prediction_rate": confidence,
            "strength":        label,
            "hold_duration":   hold,
            "regime":          regime,
        }
    except Exception:
        return None


def format_prediction(p: dict) -> str:
    emoji = "🔴" if p["mode"] == "intraday" else "🟢"
    label = "INTRADAY" if p["mode"] == "intraday" else "SWING TRADE"
    lines = [
        f"{emoji} {label} SIGNAL",
        "━" * 28,
        f"📊 Stock       : {p['symbol']}",
        f"💰 Current     : ₹{p['current_price']}",
        f"🎯 Signal      : {p['signal']}",
        f"📈 Buy At      : ₹{p['buy_at']}",
        f"🎯 Target      : ₹{p['target']}",
        f"🛑 Stop Loss   : ₹{p['stop_loss']}",
        f"⚖️  Risk/Reward : 1:{p['rr_ratio']}",
        f"🤖 Confidence  : {p['prediction_rate']}% {p['strength']}",
        f"📈 Regime      : {p.get('regime','—').upper()}",
    ]
    if p.get("hold_duration"):
        lines.append(f"⏳ Hold        : {p['hold_duration']}")
    lines.append("━" * 28)
    return "\n".join(lines)
