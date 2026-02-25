"""groq_brain.py — Groq LLM AI validation layer."""
import requests, json, warnings
warnings.filterwarnings("ignore")
from config import GROQ_API_KEY, GROQ_MODEL

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _call(messages, temperature=0.3, max_tokens=1024):
    if "your_groq" in GROQ_API_KEY:
        return "⚠️ Groq API key not configured."
    try:
        r = requests.post(GROQ_URL,
            headers={"Authorization": f"Bearer {GROQ_API_KEY}",
                     "Content-Type": "application/json"},
            json={"model": GROQ_MODEL, "messages": messages,
                  "temperature": temperature, "max_tokens": max_tokens},
            timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Groq error: {e}"


def _parse_json(text):
    try:
        s = text.find("{"); e = text.rfind("}") + 1
        if s != -1 and e > s:
            return json.loads(text[s:e])
    except Exception:
        pass
    return None


def validate_signal(pred, indicators):
    prompt = f"""Expert NSE analyst. Validate this ML signal. Respond ONLY in JSON.
Stock:{pred['symbol']} Signal:{pred['signal']} ML_Conf:{pred['prediction_rate']}%
Price:₹{pred['current_price']} Target:₹{pred['target']} SL:₹{pred['stop_loss']}
RSI:{indicators.get('rsi',50)} MACD:{indicators.get('macd',0)} ADX:{indicators.get('adx',20)}
Vol_Ratio:{indicators.get('volume_ratio',1)} Above_EMA50:{indicators.get('above_ema50',False)}
{{"validated":true/false,"adjusted_confidence":<0-100>,"final_signal":"BUY"/"SELL"/"HOLD","reasoning":"<2-3 sentences>","strength":"STRONG"/"MODERATE"/"WEAK"}}"""
    r = _call([{"role":"system","content":"NSE analyst. JSON only."},
               {"role":"user","content":prompt}], 0.1)
    return _parse_json(r) or {"validated":True,"adjusted_confidence":pred['prediction_rate'],
                               "final_signal":pred['signal'],"reasoning":r[:200],"strength":"MODERATE"}


def analyze_sentiment(symbol):
    prompt = f"""NSE analyst. Market sentiment for {symbol}. JSON only.
{{"sentiment":"BULLISH"/"BEARISH"/"NEUTRAL","score":<-100 to 100>,"summary":"<2 sentences>",
"factors":["f1","f2","f3"],"sector_outlook":"POSITIVE"/"NEGATIVE"/"NEUTRAL","risk_level":"LOW"/"MEDIUM"/"HIGH"}}"""
    r = _call([{"role":"system","content":"NSE analyst. JSON only."},
               {"role":"user","content":prompt}], 0.2)
    return _parse_json(r) or {"sentiment":"NEUTRAL","score":0,"summary":"Analysis unavailable.",
                               "factors":[],"sector_outlook":"NEUTRAL","risk_level":"MEDIUM"}


def explain_signal(pred, indicators):
    prompt = f"""Friendly teacher. Explain simply why {pred['symbol']} shows {pred['signal']}.
Price:₹{pred['current_price']} RSI:{indicators.get('rsi',50)} MACD:{indicators.get('macd',0)}
4-5 simple sentences. Start with "📊 Why {pred['signal']}?"."""
    return _call([{"role":"system","content":"Friendly stock teacher. Simple language."},
                  {"role":"user","content":prompt}], 0.4, 300)


def analyze_risk(pred):
    gain_pct = round(abs(pred['target']-pred['current_price'])/pred['current_price']*100,2)
    loss_pct = round(abs(pred['current_price']-pred['stop_loss'])/pred['current_price']*100,2)
    prompt   = f"""Risk expert. Analyse {pred['mode']} trade. JSON only.
{pred['symbol']} {pred['signal']} Entry:₹{pred['current_price']} Target:+{gain_pct}% SL:-{loss_pct}% RR:1:{pred['rr_ratio']}
{{"risk_level":"LOW"/"MEDIUM"/"HIGH","risk_score":<1-10>,"risks":["r1","r2","r3"],
"rewards":["rw1","rw2"],"verdict":"<1-2 sentences>","position_size_advice":"<advice>"}}"""
    r = _call([{"role":"system","content":"Risk expert. JSON only."},
               {"role":"user","content":prompt}], 0.2)
    return _parse_json(r) or {"risk_level":"MEDIUM","risk_score":5,"risks":["Market volatility"],
                               "rewards":["Good R:R"],"verdict":r[:150],"position_size_advice":"Risk 1-2% max."}


def full_ai_analysis(pred, indicators):
    v  = validate_signal(pred, indicators)
    s  = analyze_sentiment(pred["symbol"])
    ex = explain_signal(pred, indicators)
    r  = analyze_risk(pred)
    ml = pred["prediction_rate"]
    ai = v.get("adjusted_confidence", ml)
    sb = s.get("score", 0) * 0.1
    fc = round(max(0, min(100, ml*0.5 + ai*0.3 + (ml+sb)*0.2)), 1)
    return {"symbol":pred["symbol"],"ml_confidence":ml,"ai_confidence":ai,
            "final_confidence":fc,"final_signal":v.get("final_signal",pred["signal"]),
            "validation":v,"sentiment":s,"explanation":ex,"risk":r}


def chat_about_stock(symbol, question, history=None):
    system = f"""Expert NSE analyst for {symbol}. Concise 3-5 sentences. Educational only, not financial advice."""
    msgs   = [{"role":"system","content":system}]
    if history:
        msgs.extend(history[-6:])
    msgs.append({"role":"user","content":question})
    return _call(msgs, 0.5, 400)
