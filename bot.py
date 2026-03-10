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
    "XAUUSD": {"yahoo": "GC=F",    "min": 3000, "max": 8000, "name": "XAU/USD (Gold)"},
    "USDJPY": {"yahoo": "JPY=X",   "min": 100,  "max": 200,  "name": "USD/JPY"},
    "EURUSD": {"yahoo": "EURUSD=X","min": 0.9,  "max": 1.5,  "name": "EUR/USD"},
    "GBPUSD": {"yahoo": "GBPUSD=X","min": 1.0,  "max": 1.8,  "name": "GBP/USD"},
}

last_signal_time = {p: None for p in PAIRS}
MIN_SIGNAL_GAP = 90  # minutes between signals per pair
is_paused = False

# ── WIN/LOSS TRACKING ──────────────────────────────────────────────────────
trade_log = []  # list of dicts: {pair, direction, entry, sl, tp1, time, result}
pending_checks = []  # signals waiting for outcome check


# ── FETCH CANDLES ──────────────────────────────────────────────────────────
def fetch_candles(yahoo_symbol, interval="5m", period="2d"):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?interval={interval}&range={period}"
    r = requests.get(url, headers=headers, timeout=10)
    result = r.json()["chart"]["result"][0]
    timestamps = result["timestamp"]
    q = result["indicators"]["quote"][0]
    df = pd.DataFrame({
        "time":   pd.to_datetime(timestamps, unit="s", utc=True),
        "open":   q["open"], "high": q["high"],
        "low":    q["low"],  "close": q["close"],
        "volume": q["volume"],
    }).dropna()
    return df


def get_live_price(yahoo_symbol):
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
    r = requests.get(url, headers=headers, timeout=8)
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
    return tr.rolling(period).mean().iloc[-1]


def calc_support_resistance(df, lookback=60):
    recent = df.tail(lookback)
    highs = recent["high"].nlargest(3).round(5).tolist()
    lows = recent["low"].nsmallest(3).round(5).tolist()
    return sorted(lows), sorted(highs, reverse=True)


def get_indicators(df):
    close = df["close"]
    rsi = calc_rsi(close).iloc[-1]
    ema9  = calc_ema(close, 9)
    ema21 = calc_ema(close, 21)
    ema50 = calc_ema(close, 50)
    supports, resistances = calc_support_resistance(df)
    vol_ratio = df["volume"].iloc[-1] / df["volume"].tail(20).mean()
    atr = calc_atr(df)
    return {
        "price":         round(close.iloc[-1], 5),
        "rsi":           round(rsi, 2),
        "ema9":          round(ema9.iloc[-1], 5),
        "ema21":         round(ema21.iloc[-1], 5),
        "ema50":         round(ema50.iloc[-1], 5),
        "bullish_cross": ema9.iloc[-2] <= ema21.iloc[-2] and ema9.iloc[-1] > ema21.iloc[-1],
        "bearish_cross": ema9.iloc[-2] >= ema21.iloc[-2] and ema9.iloc[-1] < ema21.iloc[-1],
        "ema_trend":     "bullish" if ema9.iloc[-1] > ema21.iloc[-1] > ema50.iloc[-1] else
                         "bearish" if ema9.iloc[-1] < ema21.iloc[-1] < ema50.iloc[-1] else "mixed",
        "vol_ratio":     round(vol_ratio, 2),
        "high_volume":   vol_ratio > 1.3,
        "supports":      supports,
        "resistances":   resistances,
        "atr":           round(atr, 5),
    }


# ── MULTI-TIMEFRAME CONFIRMATION ───────────────────────────────────────────
def get_htf_trend(yahoo_symbol):
    try:
        df_15 = fetch_candles(yahoo_symbol, interval="15m", period="5d")
        df_1h = fetch_candles(yahoo_symbol, interval="60m", period="30d")
        close_15 = df_15["close"]
        close_1h = df_1h["close"]
        ema9_15  = calc_ema(close_15, 9).iloc[-1]
        ema21_15 = calc_ema(close_15, 21).iloc[-1]
        ema9_1h  = calc_ema(close_1h, 9).iloc[-1]
        ema21_1h = calc_ema(close_1h, 21).iloc[-1]
        rsi_15   = calc_rsi(close_15).iloc[-1]
        rsi_1h   = calc_rsi(close_1h).iloc[-1]
        trend_15 = "bullish" if ema9_15 > ema21_15 else "bearish"
        trend_1h = "bullish" if ema9_1h > ema21_1h else "bearish"
        return {
            "trend_15m": trend_15,
            "trend_1h":  trend_1h,
            "rsi_15m":   round(rsi_15, 1),
            "rsi_1h":    round(rsi_1h, 1),
            "aligned":   trend_15 == trend_1h,
            "direction": trend_1h,
        }
    except Exception as e:
        log.warning("HTF trend failed: " + str(e))
        return None


# ── NEWS FILTER ────────────────────────────────────────────────────────────
HIGH_IMPACT_HOURS = {
    "Mon": [],
    "Tue": [13, 14],
    "Wed": [13, 14, 18, 19],
    "Thu": [12, 13, 14, 18, 19],
    "Fri": [12, 13, 14, 15],
}

def is_news_time():
    now = datetime.now(timezone.utc)
    day = now.strftime("%a")
    hour = now.hour
    risky_hours = HIGH_IMPACT_HOURS.get(day, [])
    if hour in risky_hours:
        return True, "High-impact news window (avoid trading)"
    # Also block first 15 min of each hour (common news release time)
    if now.minute < 15 and hour in [8, 12, 13, 14, 15, 18]:
        return True, "Possible news release in first 15 min of hour"
    return False, ""


# ── SETUP DETECTION ────────────────────────────────────────────────────────
def detect_setup(ind, htf):
    score = 0
    reasons = []

    rsi = ind["rsi"]
    if rsi < 35:
        score += 2; reasons.append("RSI oversold (" + str(rsi) + ")")
    elif rsi > 65:
        score -= 2; reasons.append("RSI overbought (" + str(rsi) + ")")

    if ind["bullish_cross"]:
        score += 3; reasons.append("Fresh bullish EMA 9/21 cross")
    elif ind["bearish_cross"]:
        score -= 3; reasons.append("Fresh bearish EMA 9/21 cross")

    if ind["ema_trend"] == "bullish":
        score += 2; reasons.append("5m EMA trend: bullish")
    elif ind["ema_trend"] == "bearish":
        score -= 2; reasons.append("5m EMA trend: bearish")

    if ind["high_volume"]:
        adj = 1 if score > 0 else -1
        score += adj
        reasons.append("Volume spike (" + str(ind["vol_ratio"]) + "x avg)")

    # Multi-timeframe confirmation bonus
    if htf:
        if htf["aligned"]:
            bonus = 2 if htf["direction"] == "bullish" else -2
            score += bonus
            reasons.append("MTF aligned: 15m + 1H both " + htf["direction"])
        else:
            score = int(score * 0.5)
            reasons.append("MTF conflict: 15m=" + htf["trend_15m"] + " 1H=" + htf["trend_1h"] + " (score halved)")

    p = ind["price"]
    for sup in ind["supports"]:
        if abs(p - sup) / p < 0.002:
            score += 1; reasons.append("Price at support $" + str(sup))
    for res in ind["resistances"]:
        if abs(p - res) / p < 0.002:
            score -= 1; reasons.append("Price at resistance $" + str(res))

    direction = "BUY" if score >= 5 else "SELL" if score <= -5 else "NEUTRAL"
    confidence = min(95, 50 + abs(score) * 7)
    return {
        "direction": direction,
        "score": score,
        "confidence": confidence,
        "reasons": reasons,
        "is_good_setup": abs(score) >= 5,
    }


# ── SIGNAL BUILDER ─────────────────────────────────────────────────────────
def build_signal(pair, ind, setup, htf):
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

    try:
        htf_info = ""
        if htf:
            htf_info = "15m trend: " + htf["trend_15m"] + " (RSI " + str(htf["rsi_15m"]) + "), 1H trend: " + htf["trend_1h"] + " (RSI " + str(htf["rsi_1h"]) + ")"

        prompt = (
            "You are a professional " + pair + " scalping analyst. "
            "Write a 2 sentence technical summary and 1 sentence market sentiment based on:\n"
            "Price: " + str(price) + " | RSI(5m): " + str(ind["rsi"]) + " | EMA trend: " + ind["ema_trend"] + "\n"
            "Volume: " + str(ind["vol_ratio"]) + "x | ATR: " + str(atr) + "\n"
            + htf_info + "\n"
            "Signal: " + d + " | Reasons: " + ", ".join(setup["reasons"]) + "\n"
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
        commentary = {"technicalSummary": "Signal based on RSI, EMA and volume analysis.", "sentiment": "Market conditions support this setup."}

    now = datetime.now(timezone.utc)
    hour = now.hour
    session = "London" if 6<=hour<12 else "Overlap" if 12<=hour<15 else "New York" if 15<=hour<21 else "Asian"

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
        "vol_ratio": ind["vol_ratio"], "atr": atr,
        "htf_15m": htf["trend_15m"] if htf else "N/A",
        "htf_1h":  htf["trend_1h"]  if htf else "N/A",
        "time": now,
    }


def fmt_signal(s):
    d = s["signal"]
    c = s["confidence"]
    bar = "#"*(c//10) + "-"*(10-c//10)
    sup = " | ".join("$"+str(v) for v in s["supports"])
    res = " | ".join("$"+str(v) for v in s["resistances"])
    reasons = "\n".join("  + " + r for r in s["setupReasons"])
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    lines = [
        "SIGNAL: " + s["pair"] + " " + d,
        "Timeframe: " + s["timeframe"] + " | Session: " + s["sessionContext"],
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
        "RSI: " + str(s["rsi"]) + " | EMA: " + s["ema_trend"] + " | Vol: " + str(s["vol_ratio"]) + "x | ATR: " + str(s["atr"]),
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


# ── WIN/LOSS CHECKER ───────────────────────────────────────────────────────
async def check_pending_outcomes(bot):
    now = datetime.now(timezone.utc)
    still_pending = []
    for trade in pending_checks:
        age_min = (now - trade["time"]).total_seconds() / 60
        if age_min < 30:
            still_pending.append(trade)
            continue
        try:
            current = get_live_price(PAIRS[trade["pair"]]["yahoo"])
            d = trade["direction"]
            hit_tp1 = (d == "BUY" and current >= trade["tp1"]) or (d == "SELL" and current <= trade["tp1"])
            hit_sl  = (d == "BUY" and current <= trade["sl"])  or (d == "SELL" and current >= trade["sl"])
            result = "WIN" if hit_tp1 else "LOSS" if hit_sl else "OPEN"
            trade["result"] = result
            trade_log.append(trade)
            if result != "OPEN":
                emoji = "WIN" if result == "WIN" else "LOSS"
                msg = (
                    "TRADE RESULT: " + emoji + "\n"
                    + trade["pair"] + " " + d + " @ $" + str(trade["entry"]) + "\n"
                    + "Current: $" + str(round(current, 5)) + "\n"
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
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Daily Report: No completed trades today.")
        return

    today = datetime.now(timezone.utc).date()
    todays = [t for t in trade_log if t["time"].date() == today]
    wins   = sum(1 for t in todays if t.get("result") == "WIN")
    losses = sum(1 for t in todays if t.get("result") == "LOSS")
    total  = wins + losses
    rate   = round((wins / total * 100), 1) if total > 0 else 0

    by_pair = {}
    for t in todays:
        p = t["pair"]
        if p not in by_pair:
            by_pair[p] = {"wins": 0, "losses": 0}
        if t.get("result") == "WIN":
            by_pair[p]["wins"] += 1
        elif t.get("result") == "LOSS":
            by_pair[p]["losses"] += 1

    lines = [
        "DAILY PERFORMANCE REPORT",
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "====================",
        "Total Signals: " + str(total),
        "Wins:   " + str(wins),
        "Losses: " + str(losses),
        "Win Rate: " + str(rate) + "%",
        "====================",
    ]
    for pair, stats in by_pair.items():
        lines.append(pair + ": " + str(stats["wins"]) + "W / " + str(stats["losses"]) + "L")

    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="\n".join(lines))


# ── TELEGRAM COMMANDS ──────────────────────────────────────────────────────
async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    lines = ["MARKET STATUS", datetime.now(timezone.utc).strftime("%H:%M UTC"), "===================="]
    for pair, cfg in PAIRS.items():
        try:
            df = fetch_candles(cfg["yahoo"])
            price = get_live_price(cfg["yahoo"])
            df.at[df.index[-1], "close"] = price
            ind = get_indicators(df)
            lines.append(cfg["name"])
            lines.append("  Price: $" + str(ind["price"]))
            lines.append("  RSI:   " + str(ind["rsi"]))
            lines.append("  EMA:   " + ind["ema_trend"])
            lines.append("  Vol:   " + str(ind["vol_ratio"]) + "x")
            lines.append("")
        except Exception as e:
            lines.append(cfg["name"] + ": error - " + str(e))
    await update.message.reply_text("\n".join(lines))


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    is_paused = True
    await update.message.reply_text("Bot paused. Send /resume to restart signals.")


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global is_paused
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    is_paused = False
    await update.message.reply_text("Bot resumed. Scanning for setups...")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    bot = Bot(token=TELEGRAM_TOKEN)
    await send_daily_report(bot)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(TELEGRAM_CHAT_ID):
        return
    msg = (
        "AVAILABLE COMMANDS\n"
        "====================\n"
        "/status  - Live price, RSI, EMA for all pairs\n"
        "/pause   - Stop sending signals\n"
        "/resume  - Resume signals\n"
        "/report  - View today's win/loss report\n"
        "/help    - Show this message"
    )
    await update.message.reply_text(msg)


# ── SCAN JOB ───────────────────────────────────────────────────────────────
async def scan_all():
    global is_paused
    if is_paused:
        log.info("Bot is paused, skipping scan")
        return

    news_blocked, news_reason = is_news_time()
    if news_blocked:
        log.info("News filter active: " + news_reason)
        return

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
            ind = get_indicators(df)
            htf = get_htf_trend(cfg["yahoo"])
            setup = detect_setup(ind, htf)

            log.info(pair + " | $" + str(ind["price"]) + " | RSI:" + str(ind["rsi"]) + " | Score:" + str(setup["score"]) + " | Setup:" + str(setup["is_good_setup"]))

            if not setup["is_good_setup"]:
                continue

            now = datetime.now(timezone.utc)
            if last_signal_time[pair]:
                gap = (now - last_signal_time[pair]).total_seconds() / 60
                if gap < MIN_SIGNAL_GAP:
                    log.info(pair + ": too soon (" + str(round(gap)) + " min ago)"); continue

            signal = build_signal(pair, ind, setup, htf)
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=fmt_signal(signal))
            last_signal_time[pair] = now

            pending_checks.append({
                "pair": pair, "direction": signal["signal"],
                "entry": signal["entry"], "sl": signal["stopLoss"],
                "tp1": signal["takeProfit1"], "time": now, "result": None
            })

            await asyncio.sleep(3)

        except Exception as e:
            log.error(pair + " scan error: " + str(e))


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

    log.info("Bot running. Commands: /status /pause /resume /report /help")
    while True:
        await asyncio.sleep(60)


asyncio.run(main())
