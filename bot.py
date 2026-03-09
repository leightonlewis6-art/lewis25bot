import os
import json
import asyncio
import logging
import requests
from datetime import datetime
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
    try:
        r = requests.get("https://api.metals.live/v1/spot/gold", timeout=8)
        data = r.json()
        price = data[0].get("price") if isinstance(data, list) else data.get("price")
        return f"${float(price):.2f}" if price else "~$2650"
    except Exception:
        return "~$2650"


PROMPT = """You are a XAU/USD scalping expert. Return ONLY valid JSON, no extra text. Example format:
{"signal":"BUY","timeframe":"5m","entry":2645.30,"stopLoss":2638.50,"takeProfit1":2652.10,"takeProfit2":2658.80,"takeProfit3":2665.00,"riskReward":"1:2.8","confidence":82,"technicalSummary":"RSI 42 oversold recovery, EMA cross bullish.","sentiment":"Gold bid on USD weakness.","keyLevels":{"support":[2638.50,2631.00],"resistance":[2655.00,2668.00]},"invalidationLevel":2635.00,"sessionContext":"London"}"""


def generate_signal():
    price = get_gold_price()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": "Generate XAU/USD signal. Gold price: " + price + ". Time: " + datetime.utcnow().strftime("%H:%M UTC") + ". Return ONLY JSON."}
        ],
        max_tokens=400,
        temperature=0.7,
    )
    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    start = raw.find("{")
    end = raw.rfind("}") + 1
    return json.loads(raw[start:end])


def fmt(s):
    d = s.get("signal", "?")
    c = s.get("confidence", 0)
    bar = "#" * (c // 10) + "-" * (10 - c // 10)
    support = s.get("keyLevels", {}).get("support", [])
    resistance = s.get("keyLevels", {}).get("resistance", [])
    sup = " | ".join("$" + str(round(v, 2)) for v in support)
    res = " | ".join("$" + str(round(v, 2)) for v in resistance)
    now = datetime.utcnow().strftime("%H:%M UTC")
    lines = [
        "XAU/USD SIGNAL: " + d,
        "Timeframe: " + s.get("timeframe", "?") + " | Session: " + s.get("sessionContext", "?"),
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
        log.info("Sent: " + str(s.get("signal")) + " @ " + str(s.get("entry")))
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
