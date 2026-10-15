# Market Pulse

Scrapes stock prices and financial headlines twice every weekday, scores each stock,
and publishes a ranked dashboard on GitHub Pages. Research aid only, not financial advice.

**Dashboard:** https://absinthe-007.github.io/Market_Pulse/

Updates itself at ~9:45am and ~4:15pm ET, Monday to Friday. To refresh now: Actions tab → *Update market data* → **Run workflow**.

## Customize
- **Which stocks:** edit `scraper/tickers.txt` (one ticker per line).
- **Scoring weights:** `WEIGHTS` near the top of `scraper/fetch.py`.
- **News sources:** `MARKET_FEEDS` in `scraper/fetch.py` (any RSS feed works).
- **Schedule:** the `cron` lines in `.github/workflows/update.yml` (times are UTC).

## Run locally
    pip install -r requirements.txt
    python scraper/fetch.py
    python -m http.server -d docs 8000   # open http://localhost:8000

## Data sources
Prices: Yahoo Finance (via `yfinance`). Headlines: Yahoo Finance, CNBC, MarketWatch and
Google News RSS feeds. Sentiment: VADER, scored on headlines only.
