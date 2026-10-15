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


def history(t, tries=3):
    for i in range(tries):
        try:
            df = yf.Ticker(t).history(period="1y", interval="1d", auto_adjust=True)
            if len(df):
                return df
        except Exception as ex:
            print(f"  retry {t} ({ex})")
        time.sleep(3 * (i + 1))
    return None


def price_metrics(tickers):
    out = {}
    for t in tickers:
        try:
            df = history(t)
            time.sleep(0.5)
            if df is None:
                print(f"  no price data for {t}")
                continue
            close = df["Close"].dropna()
            vol = df["Volume"].dropna()
            if len(close) < 60:
                continue
            last = float(close.iloc[-1])
            ma50s = close.rolling(50).mean()
            ma50 = float(ma50s.iloc[-1])
            ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
            tr = pd.concat([df["High"] - df["Low"], (df["High"] - close.shift()).abs(),
                            (df["Low"] - close.shift()).abs()], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            yr = close.iloc[-252:]
            out[t] = {
                "ma50": round(ma50, 2),
                "ma200": round(ma200, 2) if ma200 else None,
                "atr": round(atr, 2),
                "high52": round(float(yr.max()), 2),
                "low52": round(float(yr.min()), 2),
                "chart": [round(float(x), 2) for x in close.iloc[-120:]],
                "chart_ma50": [round(float(x), 2) for x in ma50s.iloc[-120:]],
                "price": round(last, 2),
                "chg_1d": round((last / close.iloc[-2] - 1) * 100, 2),
                "mom_1m": round((last / close.iloc[-22] - 1) * 100, 2),
                "mom_3m": round((last / close.iloc[-64] - 1) * 100, 2),
                "vs_ma50": round((last / ma50 - 1) * 100, 2),
                "rsi": round(rsi(close), 1),
                "vol_ratio": round(float(vol.iloc[-1] / vol.rolling(20).mean().iloc[-1]), 2),
            }
        except Exception as ex:
            print(f"  price error {t}: {ex}")
    return out


PROFILE_KEYS = {"longName": "name", "sector": "sector", "industry": "industry", "category": "category",
                "quoteType": "type", "longBusinessSummary": "summary", "website": "website",
                "marketCap": "market_cap", "totalAssets": "fund_assets", "trailingPE": "pe",
                "forwardPE": "fwd_pe", "dividendYield": "div_yield", "beta": "beta",
                "fullTimeEmployees": "employees", "city": "city", "country": "country"}


def load_previous_profiles():
    try:
        old = json.loads(OUT_FILE.read_text())
        return {s["ticker"]: s.get("profile") for s in old.get("stocks", []) if s.get("profile")}
    except Exception:
        return {}


def profile(t, fallback=None):
    try:
        info = yf.Ticker(t).info or {}
        p = {v: info.get(k) for k, v in PROFILE_KEYS.items() if info.get(k) not in (None, "")}
        if p.get("name"):
            return p
    except Exception as ex:
        print(f"  profile error {t}: {ex}")
    return fallback or {}


def trade_signal(r):
    """Rule-based entry/exit read. Every rule is spelled out in 'reasons'."""
    p, ma50, ma200, rsi_v = r["price"], r["ma50"], r["ma200"], r["rsi"]
    up = ma200 is not None and p > ma200 and ma50 > ma200
    down = ma200 is not None and p < ma200 and ma50 < ma200 and p < ma50
    stretch = (p / ma50 - 1) * 100
    reasons, action = [], "Hold / wait"

    if up: reasons.append("Uptrend: price and 50-day average are both above the 200-day average.")
    elif down: reasons.append("Downtrend: price and 50-day average are both below the 200-day average.")
    elif ma200 is not None and p > ma50 and p < ma200:
        reasons.append("Recovering: back above the 50-day average but still under the 200-day; wait for it to clear the 200-day.")
    else: reasons.append("No clear trend: price and averages are mixed.")

    if rsi_v >= 75 or stretch >= 15:
        action = "Take profits"
        if rsi_v >= 75: reasons.append(f"RSI {rsi_v:.0f} is very overbought (75+); rallies this stretched often cool off.")
        if stretch >= 15: reasons.append(f"Price is {stretch:.0f}% above its 50-day average, far from support.")
    elif down or (p < ma50 and r["score"] <= 40):
        action = "Sell / avoid"
        if p < ma50: reasons.append("Price is below its 50-day average.")
        if r["score"] <= 40: reasons.append(f"Weak overall score ({r['score']:.0f}).")
    elif up and rsi_v < 35:
        action = "Buy zone"
        reasons.append(f"Oversold dip (RSI {rsi_v:.0f}) inside an uptrend.")
    elif up and 40 <= rsi_v <= 60 and -2 <= stretch <= 5 and r["score"] >= 45:
        action = "Buy zone"
        reasons.append(f"Pulled back near its 50-day average ({stretch:+.1f}%) with RSI {rsi_v:.0f}, a common support area.")
    elif up and r["score"] >= 62 and rsi_v < 70:
        action = "Buy zone"
        reasons.append(f"Strong momentum score ({r['score']:.0f}) without being overbought.")
    else:
        if up and r["score"] < 45: reasons.append(f"Trend is intact but the overall score is soft ({r['score']:.0f}); wait for momentum or news to improve.")
        elif up: reasons.append("Trend is fine, but price isn't at a good entry; wait for a pullback toward the 50-day average.")
        if rsi_v >= 70: reasons.append(f"RSI {int(rsi_v)} is overbought (70+); avoid chasing.")

    r["action"] = action
    r["reasons"] = reasons
    r["levels"] = {
        "entry_low": round(ma50 * 0.98, 2), "entry_high": round(ma50 * 1.02, 2),
        "stop": round(max(p - 2 * r["atr"], 0), 2),
        "trim": round(ma50 * 1.15, 2),
    }


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


def main():
    tickers = load_tickers()
    print(f"Fetching prices for {len(tickers)} tickers...")
    prices = price_metrics(tickers)

    prev_profiles = load_previous_profiles()
    rows = []
    for t in tickers:
        if t not in prices:
            continue
        print(f"  news {t}")
        news = parse_feed(TICKER_FEED.format(t=t), "Yahoo Finance", limit=15)
        sent = float(np.mean([n["sentiment"] for n in news])) if news else 0.0
        prof = profile(t, prev_profiles.get(t))
        rows.append({"ticker": t, **prices[t], "profile": prof, "sentiment": round(sent, 3),
                     "news_count": len(news),
                     "news": sorted(news, key=lambda n: n["time"] or 0, reverse=True)[:6]})
        time.sleep(0.4)  # be polite to the feed

    score(rows)
    for r in rows:
        trade_signal(r)
    rows.sort(key=lambda r: r["score"], reverse=True)

    market = []
    for name, url in MARKET_FEEDS.items():
        market += parse_feed(url, name, limit=15)
    market.sort(key=lambda n: n["time"] or 0, reverse=True)
    mood = round(float(np.mean([n["sentiment"] for n in market])), 3) if market else 0.0

    if not rows:
        print("No price data came back; keeping the previous data.json")
        return

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
