# 📈 TradeML — NSE Quant Research Platform

> Walk-forward ML · Calibrated signals · Regime-aware · Backtested

---

## 📁 Structure
```
trademl/
├── app.py                  ← 🚀 Start here
├── config.py               ← ⚙️  Your API keys
├── data_fetcher.py         ← NSE data (yfinance + Zerodha)
├── indicators.py           ← 20 technical indicators
├── ml_model.py             ← Walk-forward ML pipeline
├── predictor.py            ← Signal + trade levels
├── groq_brain.py           ← AI validation layer
├── scanner.py              ← Parallel NSE scanner
├── alerts.py               ← Telegram + Email + Desktop
├── zerodha_auth.py         ← Auto token management
├── quick_predict.py        ← CLI single stock test
├── backtest/
│   ├── engine.py           ← Full backtest simulation
│   ├── metrics.py          ← Sharpe, drawdown, win rate
│   └── costs.py            ← Zerodha brokerage + STT
└── templates/index.html    ← Web UI
```

---

## ⚙️ Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Fill config.py
```python
ZERODHA_API_KEY    = "xxxx"
ZERODHA_API_SECRET = "xxxx"
GROQ_API_KEY       = "xxxx"   # console.groq.com (free)
TELEGRAM_BOT_TOKEN = "xxxx"
TELEGRAM_CHAT_ID   = "xxxx"
EMAIL_SENDER       = "you@gmail.com"
EMAIL_PASSWORD     = "app_password"
```

### 3. Run
```bash
python app.py
```
Open → **http://localhost:5000**

---

## 🧠 What's Upgraded (vs previous versions)

| Feature | Old | New |
|---------|-----|-----|
| Data leakage | Present | Fixed — split first, then indicators |
| Validation | Single train/test | Walk-forward (5 folds) |
| Probabilities | Uncalibrated | Isotonic calibration |
| Labels | 2-class | 3-class (BUY/SELL/HOLD) |
| Regime | None | Auto-detected (trending/ranging/volatile) |
| Backtest | None | Full engine with costs + slippage |
| Metrics | Accuracy | Precision, F1, PnL, Sharpe, Drawdown |
| UI | Scanner + Analysis | + Backtest + Feature Importance + Trade Journal |

---

## 📊 Backtest Engine
- **Capital:** ₹1,00,000
- **Risk per trade:** 1% of capital
- **Slippage:** 0.1% per leg
- **Costs:** Zerodha brokerage + STT + NSE charges + GST + Stamp duty
- **Metrics:** Win rate, Sharpe, Sortino, Calmar, Max Drawdown, Profit Factor

---

## ⚠️ Reality Check
- Realistic win rate with free data: **52–57%**
- With 1:2 R:R, 52% win rate IS profitable
- RANGING regime has higher false signal rate — system auto-filters
- Always use stop loss regardless of confidence

---

## 🔑 Zerodha Login (Daily)
1. Click **Zerodha Login** in sidebar
2. Open login page, enter credentials + OTP
3. Copy `request_token` from redirect URL
4. Paste in Web UI → Connect ✅
