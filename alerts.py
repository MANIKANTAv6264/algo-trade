"""alerts.py — Telegram, Email, Desktop alerts."""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from config import (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
                    EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER)


def send_telegram(message):
    if "your_bot" in TELEGRAM_BOT_TOKEN:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def send_email(subject, body):
    if "your_email" in EMAIL_SENDER:
        return False
    try:
        msg = MIMEMultipart()
        msg["From"], msg["To"], msg["Subject"] = EMAIL_SENDER, EMAIL_RECEIVER, subject
        msg.attach(MIMEText(body, "plain"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(EMAIL_SENDER, EMAIL_PASSWORD)
            s.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        return True
    except Exception:
        return False


def send_desktop(title, message):
    try:
        from plyer import notification
        notification.notify(title=title, message=message[:255],
                            app_name="TradeML", timeout=10)
        return True
    except Exception:
        return False


def send_all(pred, formatted):
    send_telegram(formatted)
    send_email(f"🔔 {pred['signal']} {pred['symbol']} | {pred['prediction_rate']}%", formatted)
    send_desktop(f"📊 {pred['signal']} — {pred['symbol']}",
                 f"Buy:₹{pred['buy_at']} Target:₹{pred['target']} SL:₹{pred['stop_loss']}")
    print(formatted)
