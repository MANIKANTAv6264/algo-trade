"""scanner.py — Parallel NSE stock scanner."""
import time, schedule
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings; warnings.filterwarnings("ignore")

from data_fetcher import get_all_nse_stocks
from predictor    import analyze_stock, format_prediction
from alerts       import send_all
from config       import SCAN_INTERVAL_MINUTES


def _market_open():
    now = datetime.now()
    if now.weekday() >= 5: return False
    h, m = now.hour, now.minute
    return (h > 9 or (h==9 and m>=15)) and (h < 15 or (h==15 and m<=30))


def run_scan(mode="both"):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*50}\n🔍 SCANNING | {now}\n{'='*50}")
    stocks = get_all_nse_stocks()
    modes  = (["intraday","swing"] if _market_open() else ["swing"]) if mode=="both" else [mode]
    results = []

    for m in modes:
        print(f"\n📊 {m.upper()} scan ({len(stocks)} stocks)...")
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = {ex.submit(analyze_stock, s, m): s for s in stocks}
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                if r:
                    results.append(r)
                    print(f"  ✅ {r['symbol']} {r['signal']} {r['prediction_rate']}% [{r.get('regime','—')}]")
                if i % 20 == 0:
                    print(f"  ... {i}/{len(stocks)}")

    results.sort(key=lambda x: x["prediction_rate"], reverse=True)
    print(f"\n{'='*50}\n✅ {len(results)} signals found\n{'='*50}")
    for r in results[:10]:
        send_all(r, format_prediction(r))
        time.sleep(0.5)
    return results


def start_auto(mode="both"):
    print("🚀 AUTO SCANNER | Ctrl+C to stop")
    run_scan(mode)
    schedule.every(SCAN_INTERVAL_MINUTES).minutes.do(run_scan, mode=mode)
    schedule.every().day.at("15:35").do(run_scan, mode="swing")
    while True:
        try: schedule.run_pending(); time.sleep(30)
        except KeyboardInterrupt: print("\n⛔ Stopped."); break


if __name__ == "__main__":
    print("1.Intraday  2.Swing  3.Both  4.One-time")
    c = input("Choice: ").strip()
    {"1":lambda:start_auto("intraday"),"2":lambda:start_auto("swing"),
     "3":lambda:start_auto("both"),"4":lambda:run_scan("both")}.get(c,lambda:run_scan("both"))()
