import os
import json
import asyncio
import logging
import requests
from datetime import datetime
from groq import groq
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
OPENAI_API_KEY   = os.environ.get("OPENAI_API_KEY", "")
INTERVAL_MINUTES = int(os.environ.get("INTERVAL_MINUTES", "15"))

if not all([TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, OPENAI_API_KEY]):
    raise RuntimeError("Missing env vars - check Railway Variables tab")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

def get_gold_price():
    try:
        r = requests.get("https://api.metals.live/v1/spot/gold", timeout=8)
        data = r.json()
        price = data[0].get("price") if isinstance(data, list) else data.get("price")
        return f"${float(price):.2f}" if price else "~$2650"
    except Exception:
        return "~$2650"

PROMPT = """You are a XAU/USD scalping expert. Return ONLY valid JSON, no extra text:
{"signal":"BUY","timeframe":"5m","entry":2645.30,"stopLoss":2638.50,"takeProfit1":2652.10,"takeProfit2":2658.80,"takeProfit3":2665.00,"riskReward":"1:2.8","confidence":82,"technicalSummary":"RSI 42 oversold recovery, EMA cross bullish.","sentiment":"Gold bid on USD weakness.","keyLevels":{"support":[2638.50,2631.00],"resistance":[2655.00,2668.00]},"invalidationLevel":2635.00,"sessionContext":"London"}"""

def generate_signal():
    price = get_gold_price()
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Generate XAU/USD signal. Gold price: {price}. Time: {datetime.utcnow().strftime('%H:%M UTC')}. Return ONLY JSON."}
        ],
        max_tokens=400,
    )
    raw = r.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

def fmt(s):
    d = s.get("signal", "?")
    e = "BUY" if d == "BUY" else "SELL" if d == "SELL" else "NEUTRAL"
    c = s.get("confidence", 0)
    bar = "#" * (c // 10) + "-" * (10 - c // 10)
    sup = " | ".join(f"${v:.2f}" for v in s.get("keyLevels", {}).get("support", []))
    res = " | ".join(f"${v:.2f}" for v in s.get("keyLevels", {}).get("resistance", []))
    now = datetime.utcnow().strftime("%H:%M UTC")
    return (
        f"XAU/USD SIGNAL: {e}\n"
        f"Timeframe: {s.get('timeframe','?')} | Session: {s.get('sessionContext','?')}\n"
        f"--------------------\n"
        f"Entry:        ${s.get('entry', 0):.2f}\n"
        f"Stop Loss:    ${s.get('stopLoss', 0):.2f}\n"
        f"TP1:          ${s.get('takeProfit1', 0):.2f}\n"
        f"TP2:          ${s.get('takeProfit2', 0):.2f}\n"
        f"TP3:          ${s.get('takeProfit3', 0):.2f}\n"
        f"--------------------\n"
        f"R:R:          {s.get('riskReward', '?')}\n"
        f"Invalidation: ${s.get('invalidationLevel', 0):.2f}\n"
        f"--------------------\n"
        f"Support:    {sup}\n"
        f"Resistance: {res}\n"
        f"--------------------\n"
        f"{s.get('technicalSummary', '')}\n"
        f"{s.get('sentiment', '')}\n"
        f"--------------------\n"
        f"Confidence: [{bar}] {c}%\n"
        f"Time: {now}\n"
        f"For educational purposes only."
    )

async def job():
    bot = Bot(token=TELEGRAM_TOKEN)
    log.info("Generating signal...")
    try:
        s = generate_signal()
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=fmt(s))
        log.info(f"Sent: {s.get('signal')} @ {s.get('entry')}")
    except Exception as e:
        log.error(f"Error: {e}")
        try:
            await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=f"Error: {str(e)[:200]}")
        except Exception:
            pass

async def main():
    log.info(f"Bot starting - every {INTERVAL_MINUTES} min")
    await job()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(job, "interval", minutes=INTERVAL_MINUTES)
    scheduler.start()
    while True:
        await asyncio.sleep(60)

asyncio.run(main())
