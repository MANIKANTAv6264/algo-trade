"""
data_fetcher.py — Fetches historical and live NSE stock data.
Uses yfinance (15-min delayed free data).
Zerodha API used for real-time price when authenticated.
"""
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd

NSE_STOCKS = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
    "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
    "SUNPHARMA","TITAN","BAJFINANCE","NESTLEIND","ULTRACEMCO","WIPRO",
    "HCLTECH","ONGC","NTPC","POWERGRID","TECHM","INDUSINDBK","ADANIENT",
    "ADANIPORTS","BAJAJFINSV","COALINDIA","DIVISLAB","DRREDDY","EICHERMOT",
    "GRASIM","HEROMOTOCO","HINDALCO","JSWSTEEL","M&M","SBILIFE","TATAMOTORS",
    "TATASTEEL","TATACONSUM","UPL","CIPLA","BPCL","IOC","HDFCLIFE",
    "BRITANNIA","APOLLOHOSP","BAJAJ-AUTO","PIDILITIND","SIEMENS","ABB",
    "HAVELLS","VOLTAS","MARICO","DABUR","GODREJCP","COLPAL","BERGEPAINT",
    "AMBUJACEM","ACC","SHREECEM","TRENT","ZOMATO","IRCTC","HAL","BEL",
    "RECLTD","PFC","CANBK","BANKBARODA","PNB","UNIONBANK","FEDERALBNK",
    "IDFCFIRSTB","BANDHANBNK","RBLBANK","AUBANK","CHOLAFIN","MUTHOOTFIN",
    "MANAPPURAM","LICHSGFIN","OBEROIRLTY","PRESTIGE","GODREJPROP","DLF",
    "PAGEIND","RADICO","UBL","MAXHEALTH","FORTIS","ALKEM","LUPIN",
    "AUROPHARMA","BIOCON","GLENMARK","IPCALAB","TORNTPHARM","TATAPOWER",
    "TORNTPOWER","JSPL","SAIL","NMDC","HINDZINC","BALKRISIND","MRF",
    "APOLLOTYRE","CEATLTD","MOTHERSON","BOSCHLTD","EXIDEIND","MPHASIS",
    "LTTS","PERSISTENT","COFORGE","KPITTECH","OFSS","TATAELXSI","CYIENT",
    "INDIGO","CONCOR","GMRINFRA","ADANIGREEN","IGL","MGL","GAIL",
    "PETRONET","GSPL","GUJARATGAS","POLYCAB","KEI","DIXON","WHIRLPOOL",
    "ZYDUSLIFE","NATCOPHARM","GRANULES","LAURUS","SUNTV","ZEEL","DMART",
]


def _sym(symbol: str) -> str:
    return symbol + ".NS"


def fetch_historical_data(symbol: str, period="2y", interval="1d"):
    """Return OHLCV DataFrame or None."""
    try:
        df = yf.Ticker(_sym(symbol)).history(period=period, interval=interval)
        if df is None or df.empty or len(df) < 60:
            return None
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        # Remove timezone info for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[['open','high','low','close','volume']].copy()
    except Exception:
        return None


def fetch_intraday_data(symbol: str, interval="5m"):
    try:
        df = yf.Ticker(_sym(symbol)).history(period="5d", interval=interval)
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df[['open','high','low','close','volume']].copy()
    except Exception:
        return None


def fetch_live_price(symbol: str):
    """Returns live price — tries Zerodha first, falls back to yfinance."""
    # Try Zerodha real-time
    try:
        from zerodha_auth import get_saved_token, get_kite_client
        if get_saved_token():
            kite  = get_kite_client()
            quote = kite.quote(f"NSE:{symbol}")
            return round(float(quote[f"NSE:{symbol}"]["last_price"]), 2)
    except Exception:
        pass
    # Fallback to yfinance (15-min delayed)
    try:
        info = yf.Ticker(_sym(symbol)).fast_info
        return round(float(info.last_price), 2)
    except Exception:
        return None


def get_all_nse_stocks():
    return NSE_STOCKS
