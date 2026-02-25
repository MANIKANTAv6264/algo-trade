"""
zerodha_auth.py — Auto-manages Zerodha Access Token daily.
"""
import os, json
from datetime import date

TOKEN_FILE = "zerodha_token.json"


def get_saved_token():
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        with open(TOKEN_FILE) as f:
            data = json.load(f)
        if data.get("date") == str(date.today()):
            return data.get("access_token")
    except Exception:
        pass
    return None


def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        json.dump({"access_token": token, "date": str(date.today())}, f)


def get_login_url():
    from kiteconnect import KiteConnect
    from config import ZERODHA_API_KEY
    return KiteConnect(api_key=ZERODHA_API_KEY).login_url()


def generate_token_from_request(request_token):
    from kiteconnect import KiteConnect
    from config import ZERODHA_API_KEY, ZERODHA_API_SECRET
    kite    = KiteConnect(api_key=ZERODHA_API_KEY)
    session = kite.generate_session(request_token, api_secret=ZERODHA_API_SECRET)
    save_token(session["access_token"])
    return session["access_token"]


def get_kite_client():
    from kiteconnect import KiteConnect
    from config import ZERODHA_API_KEY
    token = get_saved_token()
    if not token:
        raise Exception("No valid token. Login via Web UI.")
    kite = KiteConnect(api_key=ZERODHA_API_KEY)
    kite.set_access_token(token)
    return kite
