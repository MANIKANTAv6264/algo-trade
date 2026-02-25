"""
quick_predict.py — Test a single stock from command line.
Usage:
    python quick_predict.py RELIANCE swing
    python quick_predict.py TCS intraday
"""
import sys
from predictor import analyze_stock, format_prediction

def run(symbol, mode="swing"):
    print(f"\n🔍 Analyzing {symbol} [{mode.upper()}]")
    print("⏳ Running walk-forward ML model...\n")
    r = analyze_stock(symbol, mode)
    if r: print(format_prediction(r))
    else: print(f"ℹ️  No strong signal for {symbol} in {mode} mode.")

if __name__ == "__main__":
    sym  = sys.argv[1].upper() if len(sys.argv) > 1 else "RELIANCE"
    mode = sys.argv[2].lower() if len(sys.argv) > 2 else "swing"
    run(sym, mode)
