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

# Gold is currently trading ~$5000-$5500 range in 2026
GOLD_PRICE_MIN = 3000
GOLD_PRICE_MAX = 8000


def get_gold_price():
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    # Source 1: Yahoo Finance GC=F futures
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
        r = requests.get(url, headers=headers, timeout=8)
        price = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        price = float(price)
        if GOLD_PRICE_MIN < price < GOLD_PRICE_MAX:
            log.info("Price from Yahoo GC=F: $" + str(price))
            return price, "Yahoo Finance"
    except Exception as e:
        log.warning("Yahoo GC=F failed: " + str(e))

    # Source 2: Yahoo Finance XAUUSD=X spot
    try:
        url = "https://query2.finance.yahoo.com/v8/finance/chart/XAUUSD=X"
        r = requests.get(url, headers=headers, timeout=8)
        price = r.json()["chart"]["result"][0]["meta"]["regularMarketPrice"]
        price = float(price)
        if GOLD_PRICE_MIN < price < GOLD_PRICE_MAX:
            log.info("Price from Yahoo XAUUSD: $" + str(price))
            return price, "Yahoo Finance Spot"
    except Exception as e:
        log.warning("Yahoo XAUUSD failed: " + str(e))

    # Source 3: Twelve Data free API
    try:
        url = "https://api.twelvedata.com/price?symbol=XAU/USD&apikey=demo"
        r = requests.get(url, timeout=8)
        price = float(r.json().get("price", 0))
        if GOLD_PRICE_MIN < price < GOLD_PRICE_MAX:
            log.info("Price from TwelveData: $" + str(price))
            return price, "TwelveData"
    except Exception as e:
        log.warning("TwelveData failed: " + str(e))

    # Source 4: goldprice.org API
    try:
        url = "https://data-asg.goldprice.org/dbXRates/USD"
        r = requests.get(url, headers=headers, timeout=8)
        data = r.json()
        price = float(data["items"][0]["xauPrice"])
        if GOLD_PRICE_MIN < price < GOLD_PRICE_MAX:
            log.info("Price from goldprice.org: $" + str(price))
            return price, "GoldPrice.org"
    except Exception as e:
        log.warning("goldprice.org failed: " + str(e))

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

    p = round(price, 2)
    return (
        "You are a professional XAU/USD scalping trader.\n\n"
        "TODAY'S LIVE GOLD PRICE: $" + str(p) + " per troy ounce (source: " + source + ")\n"
        "Current UTC time: " + now_utc.strftime("%H:%M") + " | Session: " + session + "\n\n"
        "CRITICAL RULES - you MUST follow these exactly:\n"
        "1. entry price MUST be between $" + str(round(p - 2, 2)) + " and $" + str(round(p + 2, 2)) + "\n"
        "2. For BUY: stopLoss = entry minus $8 to $12, takeProfit1 = entry plus $8, takeProfit2 = entry plus $16, takeProfit3 = entry plus $28\n"
        "3. For SELL: stopLoss = entry plus $8 to $12, takeProfit1 = entry minus $8, takeProfit2 = entry minus $16, takeProfit3 = entry minus $28\n"
        "4. support levels must be BELOW $" + str(p) + "\n"
        "5. resistance levels must be ABOVE $" + str(p) + "\n"
        "6. invalidationLevel for BUY = entry minus $15, for SELL = entry plus $15\n\n"
        "Return ONLY this JSON, no other text:\n"
        '{"signal":"BUY","timeframe":"5m","entry":' + str(p) + ',"stopLoss":0.00,"takeProfit1":0.00,"takeProfit2":0.00,"takeProfit3":0.00,"riskReward":"1:2.0","confidence":75,"technicalSummary":"write 2 sentences about current technicals","sentiment":"write 1 sentence about gold market","keyLevels":{"support":[0.00,0.00],"resistance":[0.00,0.00]},"invalidationLevel":0.00,"sessionContext":"' + session + '"}'
    )


def generate_signal():
    price, source = get_gold_price()

    if not price:
        raise ValueError("Could not fetch live gold price from any source")

    prompt = build_prompt(price, source)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the scalping signal now. Return ONLY the JSON object."}
        ],
        max_tokens=500,
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    signal = json.loads(raw[start:end])

    # Force correct entry price regardless of what AI returns
    signal["entry"] = round(price, 2)
    signal["_livePrice"] = "$" + str(round(price, 2))
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
        "Entry:        " + live,
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
        log.info("Sent: " + str(s.get("signal")) + " | Live price: " + str(s.get("_livePrice")))
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
