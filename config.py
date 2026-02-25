# ============================================================
#  config.py — Fill your credentials here before running!
# ============================================================

# ── ZERODHA API ──────────────────────────────────────────────
ZERODHA_API_KEY    = "your_api_key_here"
ZERODHA_API_SECRET = "your_api_secret_here"

# ── GROQ AI ──────────────────────────────────────────────────
GROQ_API_KEY = "your_groq_api_key_here"   # From console.groq.com
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── TELEGRAM ALERTS ──────────────────────────────────────────
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID   = "your_chat_id_here"

# ── EMAIL ALERTS ─────────────────────────────────────────────
EMAIL_SENDER   = "your_email@gmail.com"
EMAIL_PASSWORD = "your_gmail_app_password"
EMAIL_RECEIVER = "your_email@gmail.com"

# ── ML SETTINGS ──────────────────────────────────────────────
TRAIN_WINDOW        = 252    # ~1 year daily bars per fold
TEST_WINDOW         = 63     # ~3 months per fold
N_SPLITS            = 5      # walk-forward folds
BUY_THRESHOLD       = 0.02   # 2% forward return = BUY label
SELL_THRESHOLD      = -0.02  # -2% forward return = SELL label
FORWARD_PERIODS     = 5      # days ahead for label

# ── RISK SETTINGS ────────────────────────────────────────────
INITIAL_CAPITAL     = 100000  # ₹1,00,000
RISK_PER_TRADE      = 0.01    # 1% of capital per trade
SLIPPAGE_PCT        = 0.001   # 0.1% slippage
MIN_CONFIDENCE      = 60      # Minimum signal confidence %
MAX_POSITIONS       = 5       # Max concurrent positions

# ── INTRADAY SETTINGS ────────────────────────────────────────
INTRADAY_SL_PERCENT     = 0.5
INTRADAY_TARGET_PERCENT = 1.5

# ── SWING SETTINGS ───────────────────────────────────────────
SWING_SL_PERCENT        = 3.0
SWING_TARGET_PERCENT    = 8.0

# ── SCANNING ─────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES = 5
MIN_VOLUME            = 100000
