"""
backtest/metrics.py
Professional trading performance metrics.
"""
import numpy as np
import pandas as pd


class BacktestMetrics:
    def __init__(self, trades: list, equity_curve: pd.DataFrame,
                 initial_capital: float):
        self.trades          = pd.DataFrame(trades) if trades else pd.DataFrame()
        self.equity_curve    = equity_curve
        self.initial_capital = initial_capital

    def summary(self) -> dict:
        if self.trades.empty:
            return {"error": "No trades executed"}

        t    = self.trades
        wins = t[t["net_pnl"] > 0]
        loss = t[t["net_pnl"] <= 0]

        win_rate   = len(wins) / len(t)
        avg_win    = float(wins["net_pnl"].mean())  if len(wins) else 0
        avg_loss   = float(loss["net_pnl"].mean())  if len(loss) else 0
        gross_pnl  = float(wins["net_pnl"].sum())
        gross_loss = float(loss["net_pnl"].sum())
        pf         = abs(gross_pnl / gross_loss) if gross_loss != 0 else 999.0
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Sharpe ratio (annualised)
        daily_ret = self.equity_curve["capital"].pct_change().dropna()
        sharpe    = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                     if daily_ret.std() > 0 else 0.0)

        # Sortino ratio (downside deviation only)
        neg_ret   = daily_ret[daily_ret < 0]
        sortino   = ((daily_ret.mean() / neg_ret.std()) * np.sqrt(252)
                     if len(neg_ret) > 0 and neg_ret.std() > 0 else 0.0)

        # Max drawdown
        roll_max  = self.equity_curve["capital"].cummax()
        drawdown  = (self.equity_curve["capital"] - roll_max) / roll_max
        max_dd    = float(drawdown.min())

        # Calmar ratio
        total_ret = ((self.equity_curve["capital"].iloc[-1] - self.initial_capital)
                     / self.initial_capital)
        calmar    = total_ret / abs(max_dd) if max_dd != 0 else 0.0

        # Consecutive wins/losses
        results   = (t["net_pnl"] > 0).astype(int).tolist()
        max_consec_win  = self._max_consecutive(results, 1)
        max_consec_loss = self._max_consecutive(results, 0)

        return {
            "total_trades":        len(t),
            "win_rate":            round(win_rate * 100, 1),
            "avg_win":             round(avg_win, 2),
            "avg_loss":            round(avg_loss, 2),
            "profit_factor":       round(pf, 2),
            "sharpe_ratio":        round(sharpe, 2),
            "sortino_ratio":       round(sortino, 2),
            "calmar_ratio":        round(calmar, 2),
            "max_drawdown_pct":    round(max_dd * 100, 2),
            "total_return_pct":    round(total_ret * 100, 2),
            "total_costs":         round(float(t["costs"].sum()), 2),
            "expectancy":          round(expectancy, 2),
            "max_consec_wins":     max_consec_win,
            "max_consec_losses":   max_consec_loss,
            "final_capital":       round(float(self.equity_curve["capital"].iloc[-1]), 2),
        }

    def equity_curve_data(self) -> list:
        df = self.equity_curve.reset_index()
        return [{"date": str(r["date"])[:10], "capital": round(float(r["capital"]), 2)}
                for _, r in df.iterrows()]

    def drawdown_data(self) -> list:
        roll_max = self.equity_curve["capital"].cummax()
        dd       = (self.equity_curve["capital"] - roll_max) / roll_max * 100
        df       = dd.reset_index()
        return [{"date": str(r["date"])[:10], "drawdown": round(float(r["capital"]), 2)}
                for _, r in df.iterrows()]

    def trade_log(self) -> list:
        if self.trades.empty:
            return []
        return self.trades.to_dict("records")

    @staticmethod
    def _max_consecutive(lst, val):
        max_c = cur = 0
        for x in lst:
            cur = cur + 1 if x == val else 0
            max_c = max(max_c, cur)
        return max_c
