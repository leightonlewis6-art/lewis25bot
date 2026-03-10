import os
import json
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from groq import Groq
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# ── CONFIG ─────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "5"))

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GROQ_API_KEY]):
    raise RuntimeError("Missing env vars - check Railway Variables tab")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

PAIRS = {
    "XAUUSD": {"yahoo": "GC=F",     "min": 3000, "max": 8000, "name": "XAU/USD (Gold)"},
    "USDJPY": {"yahoo": "JPY=X",    "min": 100,  "max": 200,  "name": "USD/JPY"},
    "EURUSD": {"yahoo": "EURUSD=X", "min": 0.9,  "max": 1.5,  "name": "EUR/USD"},
    "GBPUSD": {"yahoo": "GBPUSD=X", "min": 1.0,  "max": 1.8,  "name": "GBP/USD"},
}

last_signal_time = {p: None for p in PAIRS}
MIN_SIGNAL_GAP = 90
is_paused = False
trade_log = []
pending_checks = []


# ── FETCH CANDLES ──────────────────────────────────────────────────────────
def fetch_candles(yahoo_symbol, interval="5m", period="2d"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={period}"
    r = requests.get(url, headers=headers, timeout=10)
    result = r.json()["chart"]["result"][0]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time":   pd.to_datetime(result["timestamp"], unit="s", utc=True),
        "open":   q["open"], "high": q["high"],
        "low":    q["low"],  "close": q["close"],
        "volume": q["volume"],
    }).dropna()
    return df


def get_live_price(yahoo_symbol):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}", headers=headers, timeout=8)
    return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])


# ── INDICATORS ─────────────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calc_adx(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm  = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    atr_vals = calc_atr(df, period)
    plus_di  = 100 * plus_dm.ewm(span=period).mean() / atr_vals
    minus_di = 100 * minus_dm.ewm(span=period).mean() / atr_vals
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
    return dx.ewm(span=period).mean().iloc[-1]


def calc_support_resistance(df, lookback=60):
    recent = df.tail(lookback)
    highs = recent["high"].nlargest(3).round(5).tolist()
    lows  = recent["low"].nsmallest(3).round(5).tolist()
    return sorted(lows), sorted(highs, reverse=True)


def calc_htf_levels(yahoo_symbol):
    try:
        df4h  = fetch_candles(yahoo_symbol, interval="60m", period="30d")
        sup4h, res4h = calc_support_resistance(df4h, lookback=100)
        return {"support_4h": sup4h[:2], "resistance_4h": res4h[:2]}
    except Exception:
        return {"support_4h": [], "resistance_4h": []}


def get_indicators(df):
    close = df["close"]
    rsi_series = calc_rsi(close)
    ema9  = calc_ema(close, 9)
    ema21 = calc_ema(close, 21)
    ema50 = calc_ema(close, 50)
    atr_series = calc_atr(df)
    avg_vol = df["volume"].tail(20).mean()
    vol_ratio = df["volume"].iloc[-1] / avg_vol if avg_vol > 0 and not np.isnan(avg_vol) else 1.0

    # RSI divergence: price makes new high/low but RSI doesn't
    price_new_high = df["close"].iloc[-1] > df["close"].iloc[-5:-1].max()
    price_new_low  = df["close"].iloc[-1] < df["close"].iloc[-5:-1].min()
    rsi_new_high   = rsi_series.iloc[-1] > rsi_series.iloc[-5:-1].max()
    rsi_new_low    = rsi_series.iloc[-1] < rsi_series.iloc[-5:-1].min()
    bearish_div = price_new_high and not rsi_new_high
    bullish_div = price_new_low  and not rsi_new_low

    return {
        "price":         round(close.iloc[-1], 5),
        "rsi":           round(rsi_series.iloc[-1], 2),
        "rsi_series":    rsi_series,
        "ema9":          round(ema9.iloc[-1], 5),
        "ema21":         round(ema21.iloc[-1], 5),
        "ema50":         round(ema50.iloc[-1], 5),
        "bullish_cross": ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1],
        "bearish_cross": ema9.iloc[-2] >= ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1],
        "ema_trend":     "bullish" if ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] else
                         "bearish" if ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1] else "mixed",
        "vol_ratio":     round(float(vol_ratio), 2),
        "high_volume":   float(vol_ratio) > 1.3,
        "supports":      calc_support_resistance(df)[0],
        "resistances":   calc_support_resistance(df)[1],
        "atr":           round(atr_series.iloc[-1], 5),
        "adx":           round(calc_adx(df), 2),
        "bullish_div":   bullish_div,
        "bearish_div":   bearish_div,
    }


# ── CANDLESTICK PATTERNS ───────────────────────────────────────────────────
def detect_candle_patterns(df):
    patterns = []
    if len(df) < 3:
        return patterns

    c  = df.iloc[-1]   # current candle
    p  = df.iloc[-2]   # previous candle
    p2 = df.iloc[-3]   # two back

    body     = abs(c["close"] - c["open"])
    rng      = c["high"] - c["low"]
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    prev_body  = abs(p["close"] - p["open"])

    if rng == 0:
        return patterns

    # Pin Bar (hammer/shooting star) - strong reversal signal
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        patterns.append(("BULLISH", "Bullish Pin Bar (hammer) - strong reversal signal"))
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        patterns.append(("BEARISH", "Bearish Pin Bar (shooting star) - strong reversal signal"))

    # Engulfing - momentum continuation/reversal
    if (c["close"] > c["open"] and p["close"] < p["open"]
            and c["open"] < p["close"] and c["close"] > p["open"]):
        patterns.append(("BULLISH", "Bullish Engulfing - buyers overwhelmed sellers"))
    if (c["close"] < c["open"] and p["close"] > p["open"]
            and c["open"] > p["close"] and c["close"] < p["open"]):
        patterns.append(("BEARISH", "Bearish Engulfing - sellers overwhelmed buyers"))

    # Doji - indecision, watch for breakout
    if body < rng * 0.1:
        patterns.append(("NEUTRAL", "Doji - market indecision, wait for breakout direction"))

    # Morning Star (bullish reversal 3-candle)
    if (p2["close"] < p2["open"] and
            abs(p["close"] - p["open"]) < abs(p2["close"] - p2["open"]) * 0.3 and
            c["close"] > c["open"] and c["close"] > (p2["open"] + p2["close"]) / 2):
        patterns.append(("BULLISH", "Morning Star - strong 3-candle bullish reversal"))

    # Evening Star (bearish reversal 3-candle)
    if (p2["close"] > p2["open"] and
            abs(p["close"] - p["open"]) < abs(p2["close"] - p2["open"]) * 0.3 and
            c["close"] < c["open"] and c["close"] < (p2["open"] + p2["close"]) / 2):
        patterns.append(("BEARISH", "Evening Star - strong 3-candle bearish reversal"))

    # Marubozu - strong momentum candle (no wicks)
    if upper_wick < rng * 0.05 and lower_wick < rng * 0.05:
        if c["close"] > c["open"]:
            patterns.append(("BULLISH", "Bullish Marubozu - strong buying momentum"))
        else:
            patterns.append(("BEARISH", "Bearish Marubozu - strong selling momentum"))

    # Inside Bar - consolidation before breakout
    if c["high"] < p["high"] and c["low"] > p["low"]:
        patterns.append(("NEUTRAL", "Inside Bar - consolidation, breakout incoming"))

    return patterns


# ── BACKTESTING ────────────────────────────────────────────────────────────
def backtest_setup(df, direction, lookback=80):
    """
    Simulate the current indicator setup on past candles.
    Returns win rate % based on how often same setup led to profitable move.
    """
    wins = 0
    total = 0
    atr_series = calc_atr(df)
    rsi_series = calc_rsi(df["close"])
    ema9_series  = calc_ema(df["close"], 9)
    ema21_series = calc_ema(df["close"], 21)

    # Test each historical candle from lookback to 10 candles ago
    for i in range(20, len(df) - 10):
        rsi_val  = rsi_series.iloc[i]
        ema9_val  = ema9_series.iloc[i]
        ema21_val = ema21_series.iloc[i]
        atr_val  = atr_series.iloc[i]
        price_at  = df["close"].iloc[i]

        if atr_val == 0 or np.isnan(atr_val):
            continue

        # Match similar setup conditions
        if direction == "BUY":
            setup_match = (rsi_val < 50 and ema9_val > ema21_val)
        else:
            setup_match = (rsi_val > 50 and ema9_val < ema21_val)

        if not setup_match:
            continue

        # Check outcome over next 5-10 candles
        future = df["close"].iloc[i+1:i+10]
        sl_dist = atr_val * 1.2
        tp_dist = atr_val * 2.0

        if direction == "BUY":
            hit_tp = any(future >= price_at + tp_dist)
            hit_sl = any(future <= price_at - sl_dist)
        else:
            hit_tp = any(future <= price_at - tp_dist)
            hit_sl = any(future >= price_at + sl_dist)

        if hit_tp or hit_sl:
            total += 1
            if hit_tp:
                wins += 1

    if total < 5:
        return None, total  # not enough data

    win_rate = round((wins / total) * 100, 1)
    return win_rate, total


# ── MULTI-TIMEFRAME ────────────────────────────────────────────────────────
def get_htf_trend(yahoo_symbol):
    try:
        df_15 = fetch_candles(yahoo_symbol, interval="15m", period="5d")
        df_1h = fetch_candles(yahoo_symbol, interval="60m", period="30d")
        ema9_15  = calc_ema(df_15["close"], 9).iloc[-1]
        ema21_15 = calc_ema(df_15["close"], 21).iloc[-1]
        ema9_1h  = calc_ema(df_1h["close"], 9).iloc[-1]
        ema21_1h = calc_ema(df_1h["close"], 21).iloc[-1]
        rsi_15   = calc_rsi(df_15["close"]).iloc[-1]
        rsi_1h   = calc_rsi(df_1h["close"]).iloc[-1]
        trend_15 = "bullish" if ema9_15 > ema21_15 else "bearish"
        trend_1h = "bullish" if ema9_1h > ema21_1h else "bearish"
        return {
            "trend_15m": trend_15, "trend_1h": trend_1h,
            "rsi_15m": round(rsi_15, 1), "rsi_1h": round(rsi_1h, 1),
            "aligned": trend_15 == trend_1h, "direction": trend_1h,
        }
    except Exception as e:
        log.warning("HTF failed: " + str(e))
        return None


# ── NEWS FILTER ────────────────────────────────────────────────────────────
HIGH_IMPACT_HOURS = {
    "Tue": [13, 14], "Wed": [13, 14, 18, 19],
    "Thu": [12, 13, 14, 18, 19], "Fri": [12, 13, 14, 15],
}

def is_news_time():
    now = datetime.now(timezone.utc)
    day = now.strftime("%a")
    hour = now.hour
    if hour in HIGH_IMPACT_HOURS.get(day, []):
        return True, "High-impact news window"
    if now.minute < 15 and hour in [8, 12, 13, 14, 15, 18]:
        return True, "Possible news release"
    return False, ""


# ── SETUP DETECTION ────────────────────────────────────────────────────────
def detect_setup(ind, htf, patterns):
    score = 0
    reasons = []

    rsi = ind["rsi"]
    if rsi < 35:
        score += 2; reasons.append("RSI oversold (" + str(rsi) + ")")
    elif rsi > 65:
        score -= 2; reasons.append("RSI overbought (" + str(rsi) + ")")

    if ind["bullish_cross"]:
        score += 3; reasons.append("Fresh bullish EMA 9/21 crossover")
    elif ind["bearish_cross"]:
        score -= 3; reasons.append("Fresh bearish EMA 9/21 crossover")

    if ind["ema_trend"] == "bullish":
        score += 2; reasons.append("5m EMA trend bullish (9>21>50)")
    elif ind["ema_trend"] == "bearish":
        score -= 2; reasons.append("5m EMA trend bearish (9<21<50)")

    if ind["adx"] > 25:
        adj = 1 if score > 0 else -1
        score += adj
        reasons.append("Strong trend (ADX " + str(ind["adx"]) + ")")
    elif ind["adx"] < 15:
        score = int(score * 0.7)
        reasons.append("Weak trend (ADX " + str(ind["adx"]) + ") - score reduced")

    if ind["high_volume"]:
        adj = 1 if score > 0 else -1
        score += adj
        reasons.append("Volume spike (" + str(ind["vol_ratio"]) + "x avg)")

    if ind["bullish_div"]:
        score += 2; reasons.append("Bullish RSI divergence detected")
    if ind["bearish_div"]:
        score -= 2; reasons.append("Bearish RSI divergence detected")

    if htf:
        if htf["aligned"]:
            bonus = 2 if htf["direction"] == "bullish" else -2
            score += bonus
            reasons.append("MTF aligned: 15m + 1H both " + htf["direction"])
        else:
            score = int(score * 0.5)
            reasons.append("MTF conflict: score halved")

    # Candlestick pattern scoring
    candle_score = 0
    for direction, desc in patterns:
        if direction == "BULLISH":
            candle_score += 2
            reasons.append("Pattern: " + desc)
        elif direction == "BEARISH":
            candle_score -= 2
            reasons.append("Pattern: " + desc)
        elif direction == "NEUTRAL":
            reasons.append("Pattern: " + desc)

    # Only add candle score if it agrees with existing bias
    if (score > 0 and candle_score > 0) or (score < 0 and candle_score < 0):
        score += candle_score
        if candle_score != 0:
            reasons.append("Candle pattern confirms signal direction")
    elif candle_score != 0:
        reasons.append("Candle pattern conflicts with indicator signal")

    p = ind["price"]
    for sup in ind["supports"]:
        if abs(p - sup) / p < 0.002:
            score += 1; reasons.append("Price at 5m support $" + str(sup))
    for res in ind["resistances"]:
        if abs(p - res) / p < 0.002:
            score -= 1; reasons.append("Price at 5m resistance $" + str(res))

    direction = "BUY" if score >= 5 else "SELL" if score <= -5 else "NEUTRAL"
    confidence = min(95, 50 + abs(score) * 6)

    return {
        "direction": direction, "score": score,
        "confidence": confidence, "reasons": reasons,
        "is_good_setup": abs(score) >= 5,
    }


# ── SIGNAL BUILDER ─────────────────────────────────────────────────────────
def build_signal(pair, ind, setup, htf, patterns, backtest_rate, backtest_trades):
    price = ind["price"]
    d = setup["direction"]
    atr = ind["atr"]
    sl_dist  = round(atr * 1.2, 5)
    tp1_dist = round(atr * 1.5, 5)
    tp2_dist = round(atr * 2.5, 5)
    tp3_dist = round(atr * 4.0, 5)

    if d == "BUY":
        sl, tp1, tp2, tp3 = price-sl_dist, price+tp1_dist, price+tp2_dist, price+tp3_dist
        inv = price - sl_dist * 1.5
    else:
        sl, tp1, tp2, tp3 = price+sl_dist, price-tp1_dist, price-tp2_dist, price-tp3_dist
        inv = price + sl_dist * 1.5

    rr = "1:" + str(round(tp2_dist / sl_dist, 1))

    pattern_names = [desc for _, desc in patterns] if patterns else ["No pattern detected"]

    try:
        prompt = (
            "You are a professional " + pair + " scalping analyst.\n"
            "Price: " + str(price) + " | RSI: " + str(ind["rsi"]) + " | EMA: " + ind["ema_trend"] + "\n"
            "ADX: " + str(ind["adx"]) + " | Volume: " + str(ind["vol_ratio"]) + "x\n"
            "Candlestick patterns: " + ", ".join(pattern_names) + "\n"
            "Signal: " + d + " | Score: " + str(setup["score"]) + "\n"
            "Write a 2 sentence technical summary and 1 sentence market sentiment.\n"
            "Return ONLY JSON: {\"technicalSummary\":\"...\",\"sentiment\":\"...\"}"
        )
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.3,
        )
        raw = resp.choices[0].message.content.replace("```json","").replace("```","").strip()
        commentary = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception:
        commentary = {"technicalSummary": "Signal based on RSI, EMA, ADX and candlestick analysis.", "sentiment": "Setup confirmed by multiple indicators."}

    now = datetime.now(timezone.utc)
    hour = now.hour
    session = "London" if 6<=hour<12 else "Overlap" if 12<=hour<15 else "New York" if 15<=hour<21 else "Asian"

    bt_str = str(backtest_rate) + "% (" + str(backtest_trades) + " trades)" if backtest_rate is not None else "Insufficient data"

    return {
        "pair": PAIRS[pair]["name"], "signal": d, "timeframe": "5m",
        "entry": round(price,5), "stopLoss": round(sl,5),
        "takeProfit1": round(tp1,5), "takeProfit2": round(tp2,5), "takeProfit3": round(tp3,5),
        "riskReward": rr, "confidence": setup["confidence"],
        "technicalSummary": commentary.get("technicalSummary",""),
        "sentiment": commentary.get("sentiment",""),
        "supports": ind["supports"], "resistances": ind["resistances"],
        "invalidationLevel": round(inv,5), "sessionContext": session,
        "setupReasons": setup["reasons"],
        "rsi": ind["rsi"], "ema_trend": ind["ema_trend"],
        "adx": ind["adx"], "vol_ratio": ind["vol_ratio"], "atr": atr,
        "htf_15m": htf["trend_15m"] if htf else "N/A",
        "htf_1h":  htf["trend_1h"]  if htf else "N/A",
        "patterns": pattern_names,
        "backtest": bt_str,
        "time": now,
    }


def fmt_signal(s):
    d = s["signal"]
    c = s["confidence"]
    bar = "#"*(c//10) + "-"*(10-c//10)
    sup = " | ".join("$"+str(v) for v in s["supports"])
    res = " | ".join("$"+str(v) for v in s["resistances"])
    reasons = "\n".join("  + " + r for r in s["setupReasons"])
    patterns = "\n".join("  ~ " + p for p in s["patterns"])
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        "SIGNAL: " + s["pair"] + " " + d,
        "Session: " + s["sessionContext"] + " | TF: " + s["timeframe"],
        "MTF: 15m=" + s["htf_15m"] + " | 1H=" + s["htf_1h"],
        "====================",
        "Entry:        $" + str(s["entry"]),
        "Stop Loss:    $" + str(s["stopLoss"]),
        "TP1:          $" + str(s["takeProfit1"]),
        "TP2:          $" + str(s["takeProfit2"]),
        "TP3:          $" + str(s["takeProfit3"]),
        "====================",
        "R:R:          " + s["riskReward"],
        "Invalidation: $" + str(s["invalidationLevel"]),
        "====================",
        "RSI: " + str(s["rsi"]) + " | ADX: " + str(s["adx"]) + " | EMA: " + s["ema_trend"],
        "Vol: " + str(s["vol_ratio"]) + "x | ATR: " + str(s["atr"]),
        "Support:    " + sup,
        "Resistance: " + res,
        "====================",
        "Candlestick Patterns:",
        patterns,
        "====================",
        "Setup Confluence:",
        reasons,
        "====================",
        "Backtest Win Rate: " + s["backtest"],
        "====================",
        s["technicalSummary"],
        s["sentiment"],
        "====================",
        "Confidence: [" + bar + "] " + str(c) + "%",
        "Time: " + now,
        "For educational purposes only."
    ]
    return "\n".join(lines)


# ── WIN/LOSS TRACKING ──────────────────────────────────────────────────────
async def check_pending_outcomes(bot):
    now = datetime.now(timezone.utc)
    still_pending = []
    for trade in pending_checks:
        age_min = (now - trade["time"]).total_seconds() / 60
        if age_min < 30:
            still_pending.append(trade); continue
        try:
            current = get_live_price(PAIRS[trade["pair"]]["yahoo"])
            d = trade["direction"]
            hit_tp = (d=="BUY" and current>=trade["tp1"]) or (d=="SELL" and current<=trade["tp1"])
            hit_sl = (d=="BUY" and current<=trade["sl"])  or (d=="SELL" and current>=trade["sl"])
            result = "WIN" if hit_tp else "LOSS" if hit_sl else "OPEN"
            trade["result"] = result
            trade_log.append(trade)
            if result != "OPEN":
                msg = (
                    "TRADE RESULT: " + result + "\n"
                    + trade["pair"] + " " + d + " @ $" + str(trade["entry"]) + "\n"
                    + "Current: $" + str(round(current,5)) + "\n"
                    + "TP1: $" + str(trade["tp1"]) + " | SL: $" + str(trade["sl"])
                )
                await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
            else:
                still_pending.append(trade)
        except Exception as e:
            log.warning("Outcome check failed: " + str(e))
            still_pending.append(trade)
    pending_checks.clear()
    pending_checks.extend(still_pending)


async def send_daily_report(bot):
    if not trade_log:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Daily Report: No completed trades yet.")
        return
    today = datetime.now(timezone.utc).date()
    todays = [t for t in trade_log if t["time"].date() == today]
    wins   = sum(1 for t in todays if t.get("result")=="WIN")
    losses = sum(1 for t in todays if t.get("result")=="LOSS")
    total  = wins + losses
    rate   = round(wins/total*100, 1) if total > 0 else 0
    by_pair = {}
    for t in todays:
        p = t["pair"]
        if p not in by_pair: by_pair[p] = {"wins":0,"losses":0}
        if t.get("result")=="WIN":   by_pair[p]["wins"] += 1
        elif t.get("result")=="LOSS": by_pair[p]["losses"] += 1
    lines = [
        "DAILY PERFORMANCE REPORT",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "====================",
        "Total: " + str(total) + " | Wins: " + str(wins) + " | Losses: " + str(losses),
        "Win Rate: " + str(rate) + "%",
        "====================",
    ]
    for pair, stats in by_pair.items():
        lines.append(pair + ": " + str(stats["wins"]) + "W / " + str(stats["losses"]) + "L")
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(lines))


# ── TELEGRAM COMMANDS ──────────────────────────────────────────────────────
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    lines = ["MARKET STATUS", datetime.now(timezone.utc).strftime("%H:%M UTC"), "===================="]
    for pair, cfg in PAIRS.items():
        try:
            df = fetch_candles(cfg["yahoo"])
            price = get_live_price(cfg["yahoo"])
            df.at[df.index[-1], "close"] = price
            ind = get_indicators(df)
            patterns = detect_candle_patterns(df)
            pat_str = ", ".join(d + ":" + desc[:20] for d, desc in patterns) if patterns else "None"
            lines += [
                cfg["name"],
                "  Price: $" + str(ind["price"]),
                "  RSI: " + str(ind["rsi"]) + " | ADX: " + str(ind["adx"]) + " | EMA: " + ind["ema_trend"],
                "  Vol: " + str(ind["vol_ratio"]) + "x",
                "  Patterns: " + pat_str, ""
            ]
        except Exception as e:
            lines.append(cfg["name"] + ": error - " + str(e))
    await update.message.reply_text("\n".join(lines))


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    is_paused = True
    await update.message.reply_text("Bot paused. Send /resume to restart.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    is_paused = False
    await update.message.reply_text("Bot resumed. Scanning for setups...")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    await send_daily_report(Bot(token=TELEGRAM_TOKEN))


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID): return
    await update.message.reply_text(
        "COMMANDS\n====================\n"
        "/status  - Live indicators for all pairs\n"
        "/pause   - Stop signals\n"
        "/resume  - Resume signals\n"
        "/report  - Win/loss report\n"
        "/help    - This message"
    )


# ── SCAN JOB ───────────────────────────────────────────────────────────────
async def scan_all():
    if is_paused:
        log.info("Paused, skipping scan"); return

    news_blocked, news_reason = is_news_time()
    if news_blocked:
        log.info("News filter: " + news_reason); return

    bot = Bot(token=TELEGRAM_TOKEN)
    await check_pending_outcomes(bot)

    for pair, cfg in PAIRS.items():
        try:
            log.info("Scanning " + pair + "...")
            df = fetch_candles(cfg["yahoo"])
            if len(df) < 60:
                log.warning(pair + ": not enough candles"); continue

            price = get_live_price(cfg["yahoo"])
            if not (cfg["min"] < price < cfg["max"]):
                log.warning(pair + ": price out of range: " + str(price)); continue

            df.at[df.index[-1], "close"] = price
            ind      = get_indicators(df)
            htf      = get_htf_trend(cfg["yahoo"])
            patterns = detect_candle_patterns(df)
            setup    = detect_setup(ind, htf, patterns)

            log.info(pair + " | $" + str(ind["price"]) + " | RSI:" + str(ind["rsi"]) +
                     " | ADX:" + str(ind["adx"]) + " | Score:" + str(setup["score"]) +
                     " | Patterns:" + str(len(patterns)))

            if not setup["is_good_setup"]:
                continue

            now = datetime.now(timezone.utc)
            if last_signal_time[pair]:
                gap = (now - last_signal_time[pair]).total_seconds() / 60
                if gap < MIN_SIGNAL_GAP:
                    log.info(pair + ": too soon (" + str(round(gap)) + "min ago)"); continue

            # Run backtest
            bt_rate, bt_trades = backtest_setup(df, setup["direction"])

            # Block signal if backtest win rate is below 45%
            if bt_rate is not None and bt_rate < 45:
                log.info(pair + ": backtest win rate too low (" + str(bt_rate) + "%), skipping")
                continue

            signal = build_signal(pair, ind, setup, htf, patterns, bt_rate, bt_trades)
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=fmt_signal(signal))
            last_signal_time[pair] = now

            pending_checks.append({
                "pair": pair, "direction": signal["signal"],
                "entry": signal["entry"], "sl": signal["stopLoss"],
                "tp1": signal["takeProfit1"], "time": now, "result": None
            })

            await asyncio.sleep(3)

        except Exception as e:
            log.error(pair + " error: " + str(e))


# ── MAIN ───────────────────────────────────────────────────────────────────
async def main():
    log.info("Bot starting - scanning every " + str(CHECK_INTERVAL) + " min")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("pause",  cmd_pause))
    app.add_handler(CommandHandler("resume", cmd_resume))
    app.add_handler(CommandHandler("report", cmd_report))
    app.add_handler(CommandHandler("help",   cmd_help))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(scan_all, "interval", minutes=CHECK_INTERVAL)
    scheduler.add_job(
        lambda: asyncio.create_task(send_daily_report(Bot(token=TELEGRAM_TOKEN))),
        "cron", hour=21, minute=0
    )
    scheduler.start()
    await scan_all()
    await app.initialize()
    await app.start()
    await app.updater.start_polling()
    log.info("Running. Commands: /status /pause /resume /report /help")
    while True:
        await asyncio.sleep(60)


asyncio.run(main())
