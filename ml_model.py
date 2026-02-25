"""
ml_model.py
Walk-forward ML pipeline with:
- No data leakage (split first, then indicators)
- 3-class classification (BUY / SELL / HOLD)
- Probability calibration (isotonic)
- Walk-forward cross-validation
- Trading-relevant metrics (not accuracy)
"""
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_score, f1_score

from indicators import add_all_indicators, FEATURE_COLUMNS
from config import (TRAIN_WINDOW, TEST_WINDOW, N_SPLITS,
                    BUY_THRESHOLD, SELL_THRESHOLD, FORWARD_PERIODS)

MODEL_DIR = "models"


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    3-class label using FORWARD return.
    Label at T uses close[T+FORWARD] — no leakage.
    Neutral zone avoids labeling marginal moves.
    """
    df   = df.copy()
    fret = df["close"].shift(-FORWARD_PERIODS) / df["close"] - 1
    df["label"] = 0  # HOLD
    df.loc[fret >  BUY_THRESHOLD,  "label"] =  1   # BUY
    df.loc[fret <  SELL_THRESHOLD, "label"] = -1   # SELL
    return df


def build_features(df_split: pd.DataFrame):
    """
    Compute features on ALREADY SPLIT data only.
    Never call on full dataset before splitting.
    """
    df   = add_all_indicators(df_split.copy())
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X    = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    return X, df


def _build_ensemble():
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=6,
        min_samples_leaf=20, random_state=42,
        class_weight="balanced", n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=4,
        learning_rate=0.03, subsample=0.8,
        random_state=42
    )
    lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    return VotingClassifier([("rf", rf), ("gb", gb), ("lr", lr)], voting="soft")


def _trading_metrics(y_true, y_pred, df_test):
    """Trading-relevant metrics — not plain accuracy."""
    mask = y_pred != 0
    if mask.sum() == 0:
        return {"precision": 0, "f1": 0, "pnl_pct": 0,
                "signal_rate": 0, "n_signals": 0}

    precision = precision_score(y_true[mask], y_pred[mask],
                                average="macro", zero_division=0)
    f1        = f1_score(y_true[mask], y_pred[mask],
                          average="macro", zero_division=0)

    # Simulated PnL
    closes = df_test["close"].values
    indices = np.where(mask)[0]
    pnl = 0.0
    for idx, pred in zip(indices, y_pred[mask]):
        ahead = idx + FORWARD_PERIODS
        if ahead < len(closes):
            ret  = (closes[ahead] - closes[idx]) / closes[idx]
            pnl += ret if pred == 1 else -ret

    return {
        "precision":   round(float(precision), 3),
        "f1":          round(float(f1), 3),
        "pnl_pct":     round(pnl * 100, 2),
        "signal_rate": round(float(mask.mean()), 3),
        "n_signals":   int(mask.sum()),
    }


def train_walk_forward(df_full: pd.DataFrame, symbol="stock", mode="swing"):
    """
    Walk-forward training — proper time-series validation.
    Returns: model, scaler, fold_metrics, feature_names
    """
    total   = len(df_full)
    models  = []
    scalers = []
    fold_metrics = []

    for i in range(N_SPLITS):
        train_end = TRAIN_WINDOW + i * TEST_WINDOW
        test_end  = train_end + TEST_WINDOW
        if test_end > total:
            break

        # SPLIT FIRST — then compute features separately
        train_raw = df_full.iloc[:train_end].copy()
        test_raw  = df_full.iloc[train_end:test_end].copy()

        # Label
        train_raw = create_labels(train_raw)
        test_raw  = create_labels(test_raw)

        # Features on each split independently
        X_train, train_ind = build_features(train_raw)
        X_test,  test_ind  = build_features(test_raw)

        y_train = train_raw.loc[X_train.index, "label"]
        y_test  = test_raw.loc[X_test.index,  "label"]

        if len(X_train) < 50 or len(y_train.unique()) < 2:
            continue

        # Scale — fit ONLY on train
        scaler      = StandardScaler()
        X_train_s   = scaler.fit_transform(X_train)
        X_test_s    = scaler.transform(X_test)  # transform only

        # Train base model
        base_model = _build_ensemble()
        base_model.fit(X_train_s, y_train)

        # Calibrate probabilities (removes overconfidence)
        calibrated = CalibratedClassifierCV(
            base_model, cv="prefit", method="isotonic"
        )
        calibrated.fit(X_train_s, y_train)

        # Evaluate with trading metrics
        y_pred   = calibrated.predict(X_test_s)
        metrics  = _trading_metrics(
            y_test.values, y_pred,
            test_raw.loc[X_test.index]
        )
        metrics["fold"] = i + 1
        fold_metrics.append(metrics)
        models.append(calibrated)
        scalers.append(scaler)

        print(f"  Fold {i+1}: precision={metrics['precision']:.2f} "
              f"f1={metrics['f1']:.2f} "
              f"pnl={metrics['pnl_pct']:.1f}% "
              f"signals={metrics['n_signals']}")

    if not models:
        return None, None, [], list(X_train.columns) if 'X_train' in dir() else []

    # Use most recent model for live prediction
    best_model  = models[-1]
    best_scaler = scalers[-1]
    feat_names  = list(X_train.columns)

    _save(best_model, best_scaler, symbol, mode)
    return best_model, best_scaler, fold_metrics, feat_names


def predict_live(df_recent: pd.DataFrame, model, scaler) -> tuple:
    """
    Live prediction using calibrated model.
    Returns: (signal, confidence_pct)
    """
    X, _ = build_features(df_recent.copy())
    if X.empty:
        return "HOLD", 50.0

    row   = scaler.transform(X.tail(1))
    proba = model.predict_proba(row)[0]
    pd_   = dict(zip(model.classes_, proba))

    buy  = pd_.get(1,  0)
    sell = pd_.get(-1, 0)
    hold = pd_.get(0,  0)

    if buy > 0.55 and buy == max(buy, sell, hold):
        return "BUY",  round(buy  * 100, 1)
    if sell > 0.55 and sell == max(buy, sell, hold):
        return "SELL", round(sell * 100, 1)
    return "HOLD", round(hold * 100, 1)


def _save(model, scaler, symbol, mode):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,  f"{MODEL_DIR}/{symbol}_{mode}_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{symbol}_{mode}_scaler.pkl")


def load(symbol, mode):
    mf = f"{MODEL_DIR}/{symbol}_{mode}_model.pkl"
    sf = f"{MODEL_DIR}/{symbol}_{mode}_scaler.pkl"
    if os.path.exists(mf) and os.path.exists(sf):
        return joblib.load(mf), joblib.load(sf)
    return None, None
