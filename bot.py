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
    sources = [
        # Source 1: metals.live
        lambda: _try_metals_live(),
        # Source 2: frankfurter (via USD/XAU conversion)
        lambda: _try_frankfurter(),
        # Source 3: coinpaprika gold-like fallback via metals-api
        lambda: _try_backup(),
    ]
    for source in sources:
        try:
            price = source()
            if price and 1500 < price < 5000:
                log.info("Gold price fetched: $" + str(price))
                return price
        except Exception as e:
            log.warning("Price source failed: " + str(e))
    log.warning("All price sources failed, using fallback")
    return None


def _try_metals_live():
    r = requests.get("https://api.metals.live/v1/spot/gold", timeout=6)
    data = r.json()
    price = data[0].get("price") if isinstance(data, list) else data.get("price")
    return float(price) if price else None


def _try_frankfurter():
    # goldapi.io free tier
    r = requests.get("https://api.gold-api.com/price/XAU", timeout=6)
    data = r.json()
    price = data.get("price")
    return float(price) if price else None


def _try_backup():
    # metalpriceapi free tier
    r = requests.get(
        "https://api.metalpriceapi.com/v1/latest?api_key=demo&base=XAU&currencies=USD",
        timeout=6
    )
    data = r.json()
    rate = data.get("rates", {}).get("USD")
    return float(rate) if rate else None


def build_prompt(price, price_source):
    price_str = "$" + str(round(price, 2)) if price else "unknown - use your best estimate around $2650"
    context = "LIVE PRICE: " + price_str + " (from " + price_source + ")"
    if price:
        sl_buy = round(price - 10, 2)
        sl_sell = round(price + 10, 2)
        tp1_buy = round(price + 8, 2)
        tp1_sell = round(price - 8, 2)
        sup1 = round(price - 15, 2)
        sup2 = round(price - 30, 2)
        res1 = round(price + 15, 2)
        res2 = round(price + 30, 2)
        context += (
            "\nBase your entry price VERY CLOSE to " + price_str +
            "\nFor BUY: entry near " + price_str + ", SL around $" + str(sl_buy) + ", TP1 around $" + str(tp1_buy) +
            "\nFor SELL: entry near " + price_str + ", SL around $" + str(sl_sell) + ", TP1 around $" + str(tp1_sell) +
            "\nKey levels near current price: support $" + str(sup1) + " / $" + str(sup2) +
            ", resistance $" + str(res1) + " / $" + str(res2)
        )

    return """You are a professional XAU/USD scalping trader. You MUST base your signal on the live price provided.

""" + context + """

RULES:
- entry price MUST be within $2 of the live price above
- stopLoss MUST be $5-$15 away from entry
- takeProfit levels must be realistic for scalping
- confidence between 60-95
- sessionContext: London (06:00-15:00 UTC), New York (12:00-21:00 UTC), Asian (22:00-06:00 UTC), Overlap (12:00-15:00 UTC)

Return ONLY this JSON with no extra text:
{"signal":"BUY","timeframe":"5m","entry":0.00,"stopLoss":0.00,"takeProfit1":0.00,"takeProfit2":0.00,"takeProfit3":0.00,"riskReward":"1:2.0","confidence":75,"technicalSummary":"write 2 sentences here","sentiment":"write 1 sentence here","keyLevels":{"support":[0.00,0.00],"resistance":[0.00,0.00]},"invalidationLevel":0.00,"sessionContext":"London"}"""


def generate_signal():
    price = get_gold_price()
    price_source = "metals.live + gold-api.com"

    prompt = build_prompt(price, price_source)
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the XAU/USD scalping signal now. Current time: " + now + ". Return ONLY JSON."}
        ],
        max_tokens=500,
        temperature=0.4,
    )

    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    signal = json.loads(raw[start:end])

    # Attach actual live price for display
    signal["_livePrice"] = "$" + str(round(price, 2)) if price else "unavailable"
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
    lines = [
        "XAU/USD SIGNAL: " + d,
        "Timeframe: " + s.get("timeframe", "?") + " | Session: " + s.get("sessionContext", "?"),
        "Live Spot Price: " + live,
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
        message = fmt(s)
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        log.info("Sent: " + str(s.get("signal")) + " @ " + str(s.get("entry")) + " (live: " + str(s.get("_livePrice")) + ")")
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
