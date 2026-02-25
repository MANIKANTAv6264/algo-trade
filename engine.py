"""
backtest/engine.py
Professional walk-forward backtesting engine.
Capital: ₹1,00,000 | Risk: 1% per trade | Slippage + STT modelled.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.costs   import IndianBrokerageCost
from backtest.metrics import BacktestMetrics
from config import INITIAL_CAPITAL, RISK_PER_TRADE, SLIPPAGE_PCT


class WalkForwardBacktester:

    def __init__(self, risk_per_trade=RISK_PER_TRADE, slippage=SLIPPAGE_PCT):
        self.risk_per_trade = risk_per_trade
        self.slippage       = slippage
        self.cost_model     = IndianBrokerageCost()
        self.trades         = []
        self.equity_curve   = []

    def run(self, df: pd.DataFrame, signals: pd.DataFrame,
            symbol: str = "", trade_type: str = "intraday") -> BacktestMetrics:
        """
        df:      OHLCV DataFrame indexed by date
        signals: DataFrame indexed by date with columns:
                 [signal, confidence, entry, target, stop_loss]
        """
        capital  = float(INITIAL_CAPITAL)
        position = None
        equity   = []

        for ts, row in df.iterrows():
            high  = float(row["high"])
            low   = float(row["low"])
            close = float(row["close"])

            # ── CHECK EXITS ──────────────────────────────────
            if position:
                exit_p, reason = None, None

                if position["dir"] == "BUY":
                    if low  <= position["stop"]:
                        exit_p, reason = position["stop"],  "STOP_LOSS"
                    elif high >= position["target"]:
                        exit_p, reason = position["target"], "TARGET"

                elif position["dir"] == "SELL":
                    if high >= position["stop"]:
                        exit_p, reason = position["stop"],  "STOP_LOSS"
                    elif low  <= position["target"]:
                        exit_p, reason = position["target"], "TARGET"

                # Force exit at end of day for intraday
                if trade_type == "intraday" and exit_p is None:
                    exit_p, reason = close, "EOD_EXIT"

                if exit_p:
                    # Slippage on exit
                    adj_exit = exit_p * (1 + self.slippage) \
                               if position["dir"] == "BUY" \
                               else exit_p * (1 - self.slippage)

                    pnl  = self._pnl(position, adj_exit)
                    cost = self.cost_model.total_cost(
                        position["entry"], adj_exit,
                        position["size"], trade_type
                    )
                    net_pnl  = pnl - cost
                    capital += net_pnl

                    self.trades.append({
                        "symbol":     symbol,
                        "entry_date": str(position["date"])[:10],
                        "exit_date":  str(ts)[:10],
                        "direction":  position["dir"],
                        "entry":      round(position["entry"], 2),
                        "exit":       round(adj_exit, 2),
                        "size":       position["size"],
                        "gross_pnl":  round(pnl, 2),
                        "costs":      round(cost, 2),
                        "net_pnl":    round(net_pnl, 2),
                        "reason":     reason,
                        "capital":    round(capital, 2),
                    })
                    position = None

            # ── CHECK ENTRY ──────────────────────────────────
            if position is None and ts in signals.index:
                sig = signals.loc[ts]
                if sig["signal"] in ("BUY", "SELL"):
                    entry   = float(sig["entry"])
                    sl      = float(sig["stop_loss"])
                    target  = float(sig["target"])
                    risk_pts = abs(entry - sl)

                    if risk_pts > 0:
                        # Slippage on entry
                        adj_entry = entry * (1 + self.slippage) \
                                    if sig["signal"] == "BUY" \
                                    else entry * (1 - self.slippage)
                        risk_amt  = capital * self.risk_per_trade
                        size      = int(risk_amt / risk_pts)
                        if size >= 1:
                            position = {
                                "dir":    sig["signal"],
                                "entry":  adj_entry,
                                "stop":   sl,
                                "target": target,
                                "size":   size,
                                "date":   ts,
                            }

            equity.append({"date": ts, "capital": capital})

        eq_df = pd.DataFrame(equity).set_index("date")
        self.equity_curve = eq_df
        return BacktestMetrics(self.trades, eq_df, INITIAL_CAPITAL)

    def _pnl(self, pos, exit_price):
        if pos["dir"] == "BUY":
            return (exit_price - pos["entry"]) * pos["size"]
        return (pos["entry"] - exit_price) * pos["size"]


def run_backtest_for_symbol(symbol: str, mode: str = "swing") -> dict:
    """
    Full backtest pipeline for one symbol.
    Trains model via walk-forward, then backtests on out-of-sample data.
    """
    from data_fetcher import fetch_historical_data
    from ml_model     import train_walk_forward, predict_live
    from indicators   import add_all_indicators, detect_regime
    from predictor    import calculate_trade_levels

    period   = "2y" if mode == "swing" else "3mo"
    interval = "1d" if mode == "swing" else "15m"
    df       = fetch_historical_data(symbol, period=period, interval=interval)

    if df is None or len(df) < 200:
        return {"error": f"Insufficient data for {symbol}"}

    # Train model
    print(f"\n📊 Training walk-forward model for {symbol}...")
    model, scaler, fold_metrics, feat_names = train_walk_forward(df, symbol, mode)

    if model is None:
        return {"error": "Model training failed"}

    # Generate signals on out-of-sample portion
    from config import TRAIN_WINDOW
    oos_df  = df.iloc[TRAIN_WINDOW:].copy()
    signals = []

    for i in range(len(oos_df)):
        window = df.iloc[:TRAIN_WINDOW + i].copy()
        if len(window) < 60:
            continue
        signal, conf = predict_live(window, model, scaler)
        if signal != "HOLD":
            row   = oos_df.iloc[i]
            price = float(row["close"])
            df_ind= add_all_indicators(window.tail(60))
            buy_at, target, sl, rr = calculate_trade_levels(df_ind, price, signal, mode)
            signals.append({
                "date":       oos_df.index[i],
                "signal":     signal,
                "confidence": conf,
                "entry":      buy_at,
                "target":     target,
                "stop_loss":  sl,
            })

    if not signals:
        return {"error": "No signals generated in backtest period"}

    sig_df     = pd.DataFrame(signals).set_index("date")
    backtester = WalkForwardBacktester()
    metrics    = backtester.run(oos_df, sig_df, symbol,
                                 "intraday" if mode == "intraday" else "delivery")
    summary    = metrics.summary()

    return {
        "symbol":       symbol,
        "mode":         mode,
        "fold_metrics": fold_metrics,
        "summary":      summary,
        "equity_curve": metrics.equity_curve_data(),
        "drawdown":     metrics.drawdown_data(),
        "trade_log":    metrics.trade_log()[-20:],  # last 20 trades
        "feature_names": feat_names,
    }
