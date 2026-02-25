"""
app.py — Flask backend server
Run: python app.py  →  http://localhost:5000
"""
import os, threading
from datetime import datetime
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import warnings; warnings.filterwarnings("ignore")

from data_fetcher  import fetch_historical_data, fetch_live_price, get_all_nse_stocks
from indicators    import add_all_indicators, get_feature_importance, FEATURE_COLUMNS
from predictor     import analyze_stock
from zerodha_auth  import get_saved_token, get_login_url, generate_token_from_request
from groq_brain    import full_ai_analysis, chat_about_stock
from config        import GROQ_API_KEY, MIN_CONFIDENCE, RISK_PER_TRADE, INITIAL_CAPITAL

app = Flask(__name__)
CORS(app)

scan_results  = []
alert_history = []
backtest_cache = {}
scan_status   = {"running":False,"progress":0,"total":0,"last_scan":None}

# ── AUTH ─────────────────────────────────────────────────────
@app.route("/api/auth/status")
def auth_status():
    return jsonify({"authenticated": get_saved_token() is not None})

@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    try:    return jsonify({"login_url": get_login_url()})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/api/auth/token", methods=["POST"])
def auth_token():
    rt = (request.json or {}).get("request_token","").strip()
    if not rt: return jsonify({"success":False,"error":"No token"})
    try:    generate_token_from_request(rt); return jsonify({"success":True})
    except Exception as e: return jsonify({"success":False,"error":str(e)})

# ── STOCK DATA ────────────────────────────────────────────────
@app.route("/api/stock/<symbol>/chart")
def chart(symbol):
    period   = request.args.get("period",  "3mo")
    interval = request.args.get("interval","1d")
    df = fetch_historical_data(symbol, period=period, interval=interval)
    if df is None: return jsonify({"error":"No data"}), 404
    df  = df.reset_index()
    col = df.columns[0]
    seen = set()
    rows = []
    for _, r in df.iterrows():
        t = str(r[col])[:10]
        if t in seen: continue
        seen.add(t)
        rows.append({"time":t,"open":round(float(r["open"]),2),
                     "high":round(float(r["high"]),2),"low":round(float(r["low"]),2),
                     "close":round(float(r["close"]),2),"volume":int(r["volume"])})
    return jsonify({"symbol":symbol,"data":rows})

@app.route("/api/stock/<symbol>/indicators")
def indicators(symbol):
    df = fetch_historical_data(symbol, period="6mo", interval="1d")
    if df is None: return jsonify({"error":"No data"}), 404
    df  = add_all_indicators(df)
    row = df.iloc[-1]
    return jsonify({
        "symbol":       symbol,
        "rsi":          round(float(row.get("rsi",50)),2),
        "macd":         round(float(row.get("macd",0)),4),
        "macd_signal":  round(float(row.get("macd_signal",0)),4),
        "adx":          round(float(row.get("adx",0)),2),
        "bb_position":  round(float(row.get("bb_position",0.5)),2),
        "volume_ratio": round(float(row.get("volume_ratio",1)),2),
        "ema50":        round(float(row.get("ema50",0)),2),
        "ema200":       round(float(row.get("ema200",0)),2),
        "above_ema50":  bool(row.get("above_ema50",0)),
        "above_ema200": bool(row.get("above_ema200",0)),
        "atr_pct":      round(float(row.get("atr_pct",0)),4),
    })

@app.route("/api/stock/<symbol>/price")
def live_price(symbol):
    return jsonify({"symbol":symbol,"price":fetch_live_price(symbol)})

@app.route("/api/stock/<symbol>/predict")
def predict_stock(symbol):
    mode   = request.args.get("mode","swing")
    result = analyze_stock(symbol, mode)
    return jsonify(result if result else {"signal":"HOLD","message":"No strong signal"})

@app.route("/api/stock/<symbol>/features")
def feature_importance(symbol):
    """Return feature importance for visualization."""
    from ml_model import load
    mode = request.args.get("mode","swing")
    model, scaler = load(symbol, mode)
    if model is None:
        return jsonify({"features":[]})
    imp = get_feature_importance(model, FEATURE_COLUMNS)
    return jsonify({"features":[{"name":n,"importance":round(float(v),4)} for n,v in imp[:15]]})

# ── SCANNER ───────────────────────────────────────────────────
@app.route("/api/scan/start", methods=["POST"])
def scan_start():
    if scan_status["running"]: return jsonify({"message":"Already running"})
    mode = (request.json or {}).get("mode","swing")
    threading.Thread(target=_bg_scan, args=(mode,), daemon=True).start()
    return jsonify({"message":"Scan started","mode":mode})

@app.route("/api/scan/status")
def scan_status_r(): return jsonify(scan_status)

@app.route("/api/scan/results")
def scan_results_r(): return jsonify({"results":scan_results,"count":len(scan_results)})

@app.route("/api/alerts/history")
def alerts_r(): return jsonify({"alerts":alert_history[-50:]})

@app.route("/api/stocks/list")
def stocks_list(): return jsonify({"stocks":get_all_nse_stocks()})

def _bg_scan(mode):
    global scan_results, scan_status
    stocks = get_all_nse_stocks()
    scan_status.update({"running":True,"progress":0,"total":len(stocks)})
    scan_results = []
    for i, sym in enumerate(stocks):
        try:
            r = analyze_stock(sym, mode)
            if r:
                scan_results.append(r)
                alert_history.append({**r,"timestamp":datetime.now().strftime("%H:%M:%S %d-%m-%Y")})
        except Exception:
            pass
        scan_status["progress"] = i + 1
    scan_results.sort(key=lambda x: x["prediction_rate"], reverse=True)
    scan_status.update({"running":False,"last_scan":datetime.now().strftime("%H:%M:%S")})

# ── BACKTEST ──────────────────────────────────────────────────
@app.route("/api/backtest/<symbol>", methods=["POST"])
def run_backtest(symbol):
    mode = (request.json or {}).get("mode","swing")
    key  = f"{symbol}_{mode}"
    if key in backtest_cache:
        return jsonify(backtest_cache[key])
    try:
        from backtest.engine import run_backtest_for_symbol
        result = run_backtest_for_symbol(symbol, mode)
        if "error" not in result:
            backtest_cache[key] = result
        return jsonify(result)
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# ── AI ────────────────────────────────────────────────────────
@app.route("/api/ai/analyze/<symbol>", methods=["POST"])
def ai_analyze(symbol):
    data = request.json or {}
    pred, inds = data.get("prediction"), data.get("indicators")
    if not pred or not inds: return jsonify({"error":"prediction+indicators required"}), 400
    try:    return jsonify(full_ai_analysis(pred, inds))
    except Exception as e: return jsonify({"error":str(e)}), 500

@app.route("/api/ai/chat/<symbol>", methods=["POST"])
def ai_chat(symbol):
    data = request.json or {}
    q    = data.get("question","").strip()
    hist = data.get("history",[])
    if not q: return jsonify({"error":"question required"}), 400
    return jsonify({"answer": chat_about_stock(symbol, q, hist)})

@app.route("/api/ai/status")
def ai_status():
    return jsonify({"configured": "your_groq" not in GROQ_API_KEY})

@app.route("/api/settings")
def get_settings():
    return jsonify({"min_confidence":MIN_CONFIDENCE,"risk_per_trade":RISK_PER_TRADE*100,
                    "initial_capital":INITIAL_CAPITAL})

@app.route("/")
def index(): return render_template("index.html")

if __name__ == "__main__":
    print("="*50)
    print("  🚀  TradeML Quant Platform")
    print("  👉  http://localhost:5000")
    print("="*50)
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
