import os
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

try:
    import yfinance as yf
except ImportError:
    yf = None

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

def fetch_company_news(ticker: str, days_back: int = 7) -> list[dict]:
    # FIX: Calculate "now" once to prevent midnight race condition mismatches
    now = datetime.today()
    end_date   = now.strftime("%Y-%m-%d")
    start_date = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker.upper(),
        "from":   start_date,
        "to":     end_date,
        "token":  FINNHUB_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json()

        cleaned = []
        for article in articles[:30]:
            headline = article.get("headline", "").strip()
            summary  = article.get("summary",  "").strip()
            if headline:
                cleaned.append({
                    "headline":  headline,
                    "summary":   summary,
                    "text":      f"{headline}. {summary}" if summary else headline,
                    "datetime":  datetime.fromtimestamp(article.get("datetime", 0)),
                    "url":       article.get("url", ""),
                    "source":    article.get("source", ""),
                })
        return cleaned

    except requests.RequestException as e:
        print(f"[data_fetcher] API error: {e}")
        return []

def fetch_news_yfinance(ticker: str) -> list[dict]:
    if not yf:
        print("[data_fetcher] yfinance is not installed.")
        return []
        
    try:
        stock = yf.Ticker(ticker)
        news  = stock.news or []
        results = []
        for item in news:
            content   = item.get("content", item) 
            title     = content.get("title", "")
            link      = content.get("canonicalUrl", {}).get("url", "") or content.get("link", "")
            publisher = content.get("provider", {}).get("displayName", "") or content.get("publisher", "")
            pub_time  = content.get("pubDate", "") or content.get("providerPublishTime", 0)

            if not title:
                continue

            try:
                dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00")) if isinstance(pub_time, str) else datetime.fromtimestamp(pub_time)
            except Exception:
                dt = datetime.now()

            results.append({
                "headline": title,
                "summary":  "",
                "text":     title,
                "datetime": dt,
                "url":      link,
                "source":   publisher,
            })
        return results
    except Exception as e:
        print(f"[data_fetcher] yfinance error: {e}")
        return []