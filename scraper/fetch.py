"""
Market Pulse scraper.
Pulls price history (Yahoo Finance via yfinance) and headlines (public RSS feeds),
scores each ticker on momentum + trend + news sentiment, and writes docs/data.json.

This is a research tool, not financial advice.
"""
import json
import time
import calendar
import datetime as dt
from pathlib import Path

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parent.parent
TICKERS_FILE = ROOT / "scraper" / "tickers.txt"
OUT_FILE = ROOT / "docs" / "data.json"

MARKET_FEEDS = {
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "CNBC Markets": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Google News: Stock market": "https://news.google.com/rss/search?q=stock+market&hl=en-US&gl=US&ceid=US:en",
}
TICKER_FEED = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
NEWS_LOOKBACK_DAYS = 7

# Score weights (sum of absolute values = 1)
WEIGHTS = {"mom_3m": 0.30, "mom_1m": 0.20, "trend": 0.15, "sentiment": 0.25, "rsi_penalty": 0.10}

analyzer = SentimentIntensityAnalyzer()


def load_tickers():
    lines = TICKERS_FILE.read_text().splitlines()
    return [l.strip().upper() for l in lines if l.strip() and not l.startswith("#")]


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = -delta.clip(upper=0).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return float((100 - 100 / (1 + rs)).iloc[-1])


def entry_time(e):
    t = e.get("published_parsed") or e.get("updated_parsed")
    return calendar.timegm(t) if t else None


def parse_feed(url, source, limit=20):
    try:
        feed = feedparser.parse(url, agent="Mozilla/5.0 MarketPulse")
    except Exception:
        return []
    cutoff = time.time() - NEWS_LOOKBACK_DAYS * 86400
    items = []
    for e in feed.entries[:limit]:
        ts = entry_time(e)
        if ts and ts < cutoff:
            continue
        title = e.get("title", "").strip()
        if not title:
            continue
        items.append({
            "title": title,
            "link": e.get("link", ""),
            "source": source,
            "time": ts,
            "sentiment": round(analyzer.polarity_scores(title)["compound"], 3),
        })
    return items


def price_metrics(tickers):
    data = yf.download(tickers, period="6mo", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if len(tickers) > 1 else data
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if len(close) < 60:
                continue
            last = float(close.iloc[-1])
            ma50 = float(close.rolling(50).mean().iloc[-1])
            out[t] = {
                "price": round(last, 2),
                "chg_1d": round((last / close.iloc[-2] - 1) * 100, 2),
                "mom_1m": round((last / close.iloc[-22] - 1) * 100, 2),
                "mom_3m": round((last / close.iloc[-64] - 1) * 100, 2) if len(close) > 64 else None,
                "vs_ma50": round((last / ma50 - 1) * 100, 2),
                "rsi": round(rsi(close), 1),
                "vol_ratio": round(float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]), 2),
                "spark": [round(float(x), 2) for x in close.iloc[-60:]],
            }
        except Exception as ex:
            print(f"  price error {t}: {ex}")
    return out


def zscores(values):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    mu, sd = np.nanmean(arr), np.nanstd(arr)
    z = (arr - mu) / sd if sd > 0 else np.zeros_like(arr)
    return np.nan_to_num(z)


def score(rows):
    if not rows:
        return
    z3 = zscores([r["mom_3m"] for r in rows])
    z1 = zscores([r["mom_1m"] for r in rows])
    zt = zscores([r["vs_ma50"] for r in rows])
    zs = zscores([r["sentiment"] for r in rows])
    for i, r in enumerate(rows):
        overbought = max(0.0, (r["rsi"] - 70) / 10)   # >70 RSI gets penalized
        oversold = max(0.0, (30 - r["rsi"]) / 10)     # <30 RSI also flagged
        raw = (WEIGHTS["mom_3m"] * z3[i] + WEIGHTS["mom_1m"] * z1[i] + WEIGHTS["trend"] * zt[i]
               + WEIGHTS["sentiment"] * zs[i] - WEIGHTS["rsi_penalty"] * (overbought + oversold))
        r["score"] = round(float(50 + 20 * raw), 1)   # centered on 50
        r["score"] = max(0.0, min(100.0, r["score"]))
        flags = []
        if r["rsi"] >= 70: flags.append("Overbought")
        if r["rsi"] <= 30: flags.append("Oversold")
        if r["vol_ratio"] >= 2: flags.append("Volume spike")
        if r["news_count"] == 0: flags.append("No recent news")
        r["flags"] = flags
        r["signal"] = "Strong" if r["score"] >= 62 else "Weak" if r["score"] <= 40 else "Neutral"


def main():
    tickers = load_tickers()
    print(f"Fetching prices for {len(tickers)} tickers...")
    prices = price_metrics(tickers)

    rows = []
    for t in tickers:
        if t not in prices:
            continue
        print(f"  news {t}")
        news = parse_feed(TICKER_FEED.format(t=t), "Yahoo Finance", limit=15)
        sent = float(np.mean([n["sentiment"] for n in news])) if news else 0.0
        rows.append({"ticker": t, **prices[t], "sentiment": round(sent, 3),
                     "news_count": len(news),
                     "news": sorted(news, key=lambda n: n["time"] or 0, reverse=True)[:6]})
        time.sleep(0.4)  # be polite to the feed

    score(rows)
    rows.sort(key=lambda r: r["score"], reverse=True)

    market = []
    for name, url in MARKET_FEEDS.items():
        market += parse_feed(url, name, limit=15)
    market.sort(key=lambda n: n["time"] or 0, reverse=True)
    mood = round(float(np.mean([n["sentiment"] for n in market])), 3) if market else 0.0

    OUT_FILE.write_text(json.dumps({
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "market_mood": mood,
        "weights": WEIGHTS,
        "stocks": rows,
        "market_news": market[:40],
    }, indent=1))
    print(f"Wrote {OUT_FILE} with {len(rows)} stocks and {len(market)} market headlines")


if __name__ == "__main__":
    main()
