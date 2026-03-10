import os
import json
import asyncio
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from groq import Groq
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CHECK_INTERVAL = int(os.environ.get("CHECK_INTERVAL", "15"))  # how often to scan in minutes

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GROQ_API_KEY]):
    raise RuntimeError("Missing env vars - check Railway Variables tab")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)

PAIRS = {
    "XAUUSD": {"yahoo": "GC=F",     "min": 3000, "max": 8000, "pip": 0.1,  "name": "XAU/USD (Gold)"},
    "USDJPY": {"yahoo": "JPY=X",    "min": 100,  "max": 200,  "pip": 0.01, "name": "USD/JPY"},
}

# Track last signal time per pair to avoid spamming
last_signal = {"XAUUSD": None, "USDJPY": None}
MIN_SIGNAL_GAP = 60  # minimum minutes between signals per pair


# ── PRICE & CANDLE DATA ────────────────────────────────────────────────────

def fetch_candles(yahoo_symbol, period="1d", interval="5m"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={period}"
    r = requests.get(url, headers=headers, timeout=10)
    data = r.json()
    result = data["chart"]["result"][0]
    timestamps = result["timestamp"]
    ohlcv = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time":   pd.to_datetime(timestamps, unit="s", utc=True),
        "open":   ohlcv["open"],
        "high":   ohlcv["high"],
        "low":    ohlcv["low"],
        "close":  ohlcv["close"],
        "volume": ohlcv["volume"],
    }).dropna()
    return df


def get_live_price(yahoo_symbol):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    r = requests.get(url, headers=headers, timeout=8)
    return float(r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"])


# ── TECHNICAL INDICATORS ───────────────────────────────────────────────────

def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_support_resistance(df, lookback=50):
    recent = df.tail(lookback)
    highs = recent["high"].nlargest(3).values.tolist()
    lows  = recent["low"].nsmallest(3).values.tolist()
    return sorted(lows), sorted(highs)


def calc_volume_signal(df):
    avg_vol = df["volume"].tail(20).mean()
    last_vol = df["volume"].iloc[-1]
    return last_vol / avg_vol if avg_vol > 0 else 1.0


def get_indicators(df):
    close = df["close"]
    rsi = calc_rsi(close).iloc[-1]
    ema9  = calc_ema(close, 9).iloc[-1]
    ema21 = calc_ema(close, 21).iloc[-1]
    ema50 = calc_ema(close, 50).iloc[-1]
    prev_ema9  = calc_ema(close, 9).iloc[-2]
    prev_ema21 = calc_ema(close, 21).iloc[-2]
    supports, resistances = calc_support_resistance(df)
    vol_ratio = calc_volume_signal(df)
    current_price = close.iloc[-1]
    prev_price    = close.iloc[-2]

    # EMA crossover signals
    bullish_cross = prev_ema9 <= prev_ema21 and ema9 > ema21
    bearish_cross = prev_ema9 >= prev_ema21 and ema9 < ema21

    return {
        "price":         round(current_price, 5),
        "rsi":           round(rsi, 2),
        "ema9":          round(ema9, 5),
        "ema21":         round(ema21, 5),
        "ema50":         round(ema50, 5),
        "bullish_cross": bullish_cross,
        "bearish_cross": bearish_cross,
        "ema_trend":     "bullish" if ema9 > ema21 > ema50 else "bearish" if ema9 < ema21 < ema50 else "mixed",
        "vol_ratio":     round(vol_ratio, 2),
        "high_volume":   vol_ratio > 1.3,
        "supports":      [round(x, 5) for x in supports],
        "resistances":   [round(x, 5) for x in resistances],
        "price_change":  round(current_price - prev_price, 5),
    }


# ── SETUP DETECTION ────────────────────────────────────────────────────────

def detect_setup(ind):
    signals = []
    score = 0

    rsi   = ind["rsi"]
    price = ind["price"]

    # RSI signals
    if rsi < 35:
        signals.append("RSI oversold (" + str(rsi) + ")")
        score += 2
    elif rsi > 65:
        signals.append("RSI overbought (" + str(rsi) + ")")
        score -= 2
    elif 40 <= rsi <= 55:
        signals.append("RSI neutral (" + str(rsi) + ")")

    # EMA crossover
    if ind["bullish_cross"]:
        signals.append("Bullish EMA9/21 crossover")
        score += 3
    elif ind["bearish_cross"]:
        signals.append("Bearish EMA9/21 crossover")
        score -= 3

    # EMA trend
    if ind["ema_trend"] == "bullish":
        signals.append("EMA trend: bullish (9>21>50)")
        score += 2
    elif ind["ema_trend"] == "bearish":
        signals.append("EMA trend: bearish (9<21<50)")
        score -= 2

    # Volume confirmation
    if ind["high_volume"]:
        signals.append("High volume confirmation (" + str(ind["vol_ratio"]) + "x avg)")
        score = score + 1 if score > 0 else score - 1

    # Support / resistance proximity
    for sup in ind["supports"]:
        if abs(price - sup) / price < 0.003:
            signals.append("Price near support $" + str(sup))
            score += 1

    for res in ind["resistances"]:
        if abs(price - res) / price < 0.003:
            signals.append("Price near resistance $" + str(res))
            score -= 1

    # Determine direction
    if score >= 4:
        direction = "BUY"
    elif score <= -4:
        direction = "SELL"
    else:
        direction = "NEUTRAL"

    confidence = min(95, 50 + abs(score) * 8)

    return {
        "direction":  direction,
        "score":      score,
        "confidence": confidence,
        "reasons":    signals,
        "is_good_setup": abs(score) >= 4,
    }


# ── SIGNAL GENERATION ─────────────────────────────────────────────────────

def generate_signal(pair, ind, setup):
    price = ind["price"]
    d     = setup["direction"]
    pip   = PAIRS[pair]["pip"]

    sl_dist  = round(price * 0.0015, 5)
    tp1_dist = round(sl_dist * 1.5, 5)
    tp2_dist = round(sl_dist * 2.5, 5)
    tp3_dist = round(sl_dist * 4.0, 5)

    if d == "BUY":
        sl  = round(price - sl_dist, 5)
        tp1 = round(price + tp1_dist, 5)
        tp2 = round(price + tp2_dist, 5)
        tp3 = round(price + tp3_dist, 5)
        inv = round(price - sl_dist * 1.5, 5)
    else:
        sl  = round(price + sl_dist, 5)
        tp1 = round(price - tp1_dist, 5)
        tp2 = round(price - tp2_dist, 5)
        tp3 = round(price - tp3_dist, 5)
        inv = round(price + sl_dist * 1.5, 5)

    rr = "1:" + str(round(tp2_dist / sl_dist, 1))

    # Ask Groq for technical & sentiment summary
    prompt = (
        "You are a professional " + pair + " scalping analyst.\n"
        "Current price: " + str(price) + "\n"
        "RSI: " + str(ind["rsi"]) + "\n"
        "EMA9: " + str(ind["ema9"]) + " EMA21: " + str(ind["ema21"]) + " EMA50: " + str(ind["ema50"]) + "\n"
        "EMA trend: " + ind["ema_trend"] + "\n"
        "Volume ratio vs average: " + str(ind["vol_ratio"]) + "x\n"
        "Setup signals: " + ", ".join(setup["reasons"]) + "\n"
        "Signal direction: " + d + "\n\n"
        "Write a 2 sentence technical summary and 1 sentence market sentiment. "
        "Return ONLY JSON: {\"technicalSummary\":\"...\",\"sentiment\":\"...\"}"
    )

    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        raw = resp.choices[0].message.content.replace("```json","").replace("```","").strip()
        commentary = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception:
        commentary = {
            "technicalSummary": "Setup detected based on RSI, EMA crossover and volume analysis.",
            "sentiment": "Signal generated from live technical indicators."
        }

    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    if 6 <= hour < 12:     session = "London"
    elif 12 <= hour < 15:  session = "Overlap"
    elif 15 <= hour < 21:  session = "New York"
    else:                  session = "Asian"

    return {
        "pair":             PAIRS[pair]["name"],
        "signal":           d,
        "timeframe":        "5m",
        "entry":            price,
        "stopLoss":         sl,
        "takeProfit1":      tp1,
        "takeProfit2":      tp2,
        "takeProfit3":      tp3,
        "riskReward":       rr,
        "confidence":       setup["confidence"],
        "technicalSummary": commentary.get("technicalSummary", ""),
        "sentiment":        commentary.get("sentiment", ""),
        "supports":         ind["supports"],
        "resistances":      ind["resistances"],
        "invalidationLevel":inv,
        "sessionContext":   session,
        "setupReasons":     setup["reasons"],
        "rsi":              ind["rsi"],
        "ema_trend":        ind["ema_trend"],
        "vol_ratio":        ind["vol_ratio"],
    }


# ── FORMAT MESSAGE ─────────────────────────────────────────────────────────

def fmt(s):
    d   = s["signal"]
    c   = s["confidence"]
    bar = "#" * (c // 10) + "-" * (10 - c // 10)
    sup = " | ".join("$" + str(v) for v in s["supports"])
    res = " | ".join("$" + str(v) for v in s["resistances"])
    reasons = "\n".join("  + " + r for r in s["setupReasons"])
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        "SIGNAL: " + s["pair"] + " " + d,
        "Timeframe: " + s["timeframe"] + " | Session: " + s["sessionContext"],
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
        "RSI: " + str(s["rsi"]) + " | EMA: " + s["ema_trend"] + " | Vol: " + str(s["vol_ratio"]) + "x",
        "Support:    " + sup,
        "Resistance: " + res,
        "====================",
        "Setup reasons:",
        reasons,
        "====================",
        s["technicalSummary"],
        s["sentiment"],
        "====================",
        "Confidence: [" + bar + "] " + str(c) + "%",
        "Time: " + now,
        "For educational purposes only."
    ]
    return "\n".join(lines)


# ── SCAN JOB ───────────────────────────────────────────────────────────────

async def scan_pair(pair, bot):
    cfg = PAIRS[pair]
    log.info("Scanning " + pair + "...")

    try:
        df = fetch_candles(cfg["yahoo"])
        if len(df) < 55:
            log.warning(pair + ": not enough candles")
            return

        price = get_live_price(cfg["yahoo"])
        if not (cfg["min"] < price < cfg["max"]):
            log.warning(pair + ": price out of valid range: " + str(price))
            return

        # Use live price as last close
        df.at[df.index[-1], "close"] = price

        ind   = get_indicators(df)
        setup = detect_setup(ind)

        log.info(pair + " | Price: $" + str(ind["price"]) + " | RSI: " + str(ind["rsi"]) + " | Score: " + str(setup["score"]) + " | Setup: " + str(setup["is_good_setup"]))

        if not setup["is_good_setup"]:
            log.info(pair + ": no good setup detected, skipping")
            return

        # Enforce minimum gap between signals
        now = datetime.now(timezone.utc)
        if last_signal[pair]:
            gap = (now - last_signal[pair]).total_seconds() / 60
            if gap < MIN_SIGNAL_GAP:
                log.info(pair + ": last signal was " + str(round(gap)) + " min ago, waiting")
                return

        signal = generate_signal(pair, ind, setup)
        message = fmt(signal)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        last_signal[pair] = now
        log.info(pair + ": signal sent - " + setup["direction"] + " @ $" + str(ind["price"]))

    except Exception as e:
        log.error(pair + " error: " + str(e))


async def scan_all():
    bot = Bot(token=TELEGRAM_TOKEN)
    for pair in PAIRS:
        await scan_pair(pair, bot)
        await asyncio.sleep(2)


async def main():
    log.info("Bot starting - scanning every " + str(CHECK_INTERVAL) + " min for good setups")
    await scan_all()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scan_all, "interval", minutes=CHECK_INTERVAL)
    scheduler.start()
    log.info("Scheduler running")
    while True:
        await asyncio.sleep(60)


asyncio.run(main())
