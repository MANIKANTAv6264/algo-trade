"""
backtest/costs.py
Accurate Indian market transaction cost model (Zerodha).
"""


class IndianBrokerageCost:
    """
    Zerodha cost model for NSE equity.
    Source: zerodha.com/charges
    """
    # Zerodha brokerage
    BROKERAGE_INTRADAY = 0.0003   # 0.03% or ₹20 max per leg
    BROKERAGE_DELIVERY = 0.0      # Free

    # STT (Securities Transaction Tax)
    STT_INTRADAY_SELL  = 0.00025  # 0.025% on sell turnover
    STT_DELIVERY_SELL  = 0.001    # 0.1% on sell turnover

    # Exchange charges
    NSE_TXN_CHARGE     = 0.0000297  # 0.00297% of turnover

    # Other
    GST_RATE           = 0.18    # 18% on brokerage + exchange charges
    SEBI_CHARGE        = 0.000001  # ₹10 per crore
    STAMP_DUTY_BUY     = 0.00015   # 0.015% on buy turnover

    def total_cost(self, entry_price: float, exit_price: float,
                   qty: int, trade_type: str = "intraday") -> float:
        """
        Returns total round-trip cost in ₹.
        trade_type: 'intraday' or 'delivery'
        """
        buy_turnover  = entry_price * qty
        sell_turnover = exit_price  * qty
        total_turn    = buy_turnover + sell_turnover

        # Brokerage (both legs)
        if trade_type == "intraday":
            brok = min(self.BROKERAGE_INTRADAY * buy_turnover,  20) + \
                   min(self.BROKERAGE_INTRADAY * sell_turnover, 20)
        else:
            brok = 0.0

        # STT (sell side only for intraday; sell side for delivery)
        stt = (self.STT_INTRADAY_SELL if trade_type == "intraday"
               else self.STT_DELIVERY_SELL) * sell_turnover

        # Exchange transaction charges
        txn = self.NSE_TXN_CHARGE * total_turn

        # GST on brokerage + txn
        gst = self.GST_RATE * (brok + txn)

        # SEBI
        sebi = self.SEBI_CHARGE * total_turn

        # Stamp duty on buy
        stamp = self.STAMP_DUTY_BUY * buy_turnover

        return round(brok + stt + txn + gst + sebi + stamp, 2)

    def cost_per_trade_pct(self, price: float, qty: int,
                            trade_type: str = "intraday") -> float:
        """Returns cost as % of trade value."""
        cost  = self.total_cost(price, price, qty, trade_type)
        value = price * qty
        return round((cost / value) * 100, 4) if value else 0
