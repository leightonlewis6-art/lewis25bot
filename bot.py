import os
import json
import asyncio
import logging
import requests
from datetime import datetime, timezone
from groq import Groq
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
INTERVAL_MINUTES = int(os.environ.get("INTERVAL_MINUTES", "15"))

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, GROQ_API_KEY]):
    raise RuntimeError("Missing env vars - check Railway Variables tab")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
client = Groq(api_key=GROQ_API_KEY)


def get_gold_price():
    headers = {"User-Agent": "Mozilla/5.0"}

    # Source 1: Yahoo Finance GC=F (Gold Futures)
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F?interval=1m&range=1d"
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        if price and 1500 < float(price) < 5000:
            log.info("Yahoo Finance price: $" + str(price))
            return float(price), "Yahoo Finance (GC=F)"
    except Exception as e:
        log.warning("Yahoo Finance failed: " + str(e))

    # Source 2: Yahoo Finance XAUUSD
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d"
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        price = data["chart"]["result"][0]["meta"]["regularMarketPrice"]
        if price and 1500 < float(price) < 5000:
            log.info("Yahoo XAUUSD price: $" + str(price))
            return float(price), "Yahoo Finance (XAUUSD)"
    except Exception as e:
        log.warning("Yahoo XAUUSD failed: " + str(e))

    # Source 3: metals.live
    try:
        r = requests.get("https://api.metals.live/v1/spot/gold", timeout=8)
        data = r.json()
        price = data[0].get("price") if isinstance(data, list) else data.get("price")
        if price and 1500 < float(price) < 5000:
            log.info("metals.live price: $" + str(price))
            return float(price), "metals.live"
    except Exception as e:
        log.warning("metals.live failed: " + str(e))

    log.warning("All price sources failed")
    return None, "unavailable"


def build_prompt(price, source):
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour

    if 6 <= hour < 12:
        session = "London"
    elif 12 <= hour < 15:
        session = "Overlap"
    elif 15 <= hour < 21:
        session = "New York"
    else:
        session = "Asian"

    if price:
        p = round(price, 2)
        context = (
            "LIVE GOLD PRICE RIGHT NOW: $" + str(p) + " (source: " + source + ")\n"
            "Current UTC time: " + now_utc.strftime("%H:%M") + " | Trading session: " + session + "\n\n"
            "IMPORTANT: Your entry price MUST be within $1.00 of $" + str(p) + "\n"
            "For scalping use SL of $8-$12, TP1=$8, TP2=$15, TP3=$25\n"
            "Support levels should be BELOW $" + str(p) + "\n"
            "Resistance levels should be ABOVE $" + str(p)
        )
    else:
        context = "Live price unavailable. Use your best estimate for current XAU/USD price."

    return """You are a professional XAU/USD scalping trader with 10 years experience.

""" + context + """

Generate a precise scalping signal. Return ONLY this exact JSON structure, no other text:
{"signal":"BUY","timeframe":"5m","entry":0.00,"stopLoss":0.00,"takeProfit1":0.00,"takeProfit2":0.00,"takeProfit3":0.00,"riskReward":"1:2.0","confidence":75,"technicalSummary":"2 sentences about RSI MACD EMA price action","sentiment":"1 sentence about gold market conditions","keyLevels":{"support":[0.00,0.00],"resistance":[0.00,0.00]},"invalidationLevel":0.00,"sessionContext":"London"}"""


def generate_signal():
    price, source = get_gold_price()
    prompt = build_prompt(price, source)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the signal now. Return ONLY the JSON."}
        ],
        max_tokens=500,
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    signal = json.loads(raw[start:end])
    signal["_livePrice"] = "$" + str(round(price, 2)) if price else "unavailable"
    signal["_source"] = source
    return signal


def fmt(s):
    d = s.get("signal", "?")
    c = s.get("confidence", 0)
    bar = "#" * (c // 10) + "-" * (10 - c // 10)
    support = s.get("keyLevels", {}).get("support", [])
    resistance = s.get("keyLevels", {}).get("resistance", [])
    sup = " | ".join("$" + str(round(v, 2)) for v in support)
    res = " | ".join("$" + str(round(v, 2)) for v in resistance)
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")
    live = s.get("_livePrice", "N/A")
    src = s.get("_source", "")
    lines = [
        "XAU/USD SIGNAL: " + d,
        "Timeframe: " + s.get("timeframe", "?") + " | Session: " + s.get("sessionContext", "?"),
        "Live Spot: " + live + " (" + src + ")",
        "--------------------",
        "Entry:        $" + str(round(s.get("entry", 0), 2)),
        "Stop Loss:    $" + str(round(s.get("stopLoss", 0), 2)),
        "TP1:          $" + str(round(s.get("takeProfit1", 0), 2)),
        "TP2:          $" + str(round(s.get("takeProfit2", 0), 2)),
        "TP3:          $" + str(round(s.get("takeProfit3", 0), 2)),
        "--------------------",
        "R:R:          " + s.get("riskReward", "?"),
        "Invalidation: $" + str(round(s.get("invalidationLevel", 0), 2)),
        "--------------------",
        "Support:    " + sup,
        "Resistance: " + res,
        "--------------------",
        s.get("technicalSummary", ""),
        s.get("sentiment", ""),
        "--------------------",
        "Confidence: [" + bar + "] " + str(c) + "%",
        "Time: " + now,
        "For educational purposes only."
    ]
    return "\n".join(lines)


async def job():
    bot = Bot(token=TELEGRAM_TOKEN)
    log.info("Generating signal...")
    try:
        s = generate_signal()
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=fmt(s))
        log.info("Sent: " + str(s.get("signal")) + " @ " + str(s.get("entry")) + " | Live: " + str(s.get("_livePrice")))
    except Exception as e:
        log.error("Error: " + str(e))
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="Error: " + str(e)[:200])
        except Exception:
            pass


async def main():
    log.info("Bot starting - every " + str(INTERVAL_MINUTES) + " min")
    await job()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job, "interval", minutes=INTERVAL_MINUTES)
    scheduler.start()
    while True:
        await asyncio.sleep(60)


asyncio.run(main())
