import os
import json
import asyncio
import logging
import requests
from datetime import datetime
from groq import Groq
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

TELEGRAM_TOKEN   = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
GROQ_API_KEY     = os.environ.get("GROQ_API_KEY", "")
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

PROMPT = """You are a XAU/USD scalping expert. Return ONLY valid JSON, no extra text:
{"signal":"BUY","timeframe":"5m","entry":2645.30,"stopLoss":2638.50,"takeProfit1":2652.10,"takeProfit2":2658.80,"takeProfit3":2665.00,"riskReward":"1:2.8","confidence":82,"technicalSummary":"RSI 42 oversold recovery, EMA cross bullish.","sentiment":"Gold bid on USD weakness.","keyLevels":{"support":[2638.50,2631.00],"resistance":[2655.00,2668.00]},"invalidationLevel":2635.00,"sessionContext":"London"}"""

def generate_signal():
    price = get_gold_price()
    r = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": f"Generate XAU/USD signal. Gold price: {price}. Time: {datetime.utcnow().strftime('%H:%M UTC')}. Return ONLY JSON."}
        ],
        max_tokens=400,
        temperature=0.7,
    )
    raw = r.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    return json.loads(raw[raw.find("{"):raw.rfind("}") + 1])

def fmt(s
