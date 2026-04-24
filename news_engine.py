"""Real-time news monitoring and sentiment engine for trading decisions.

A sentiment model tries to estimate whether news is generally bullish
or bearish for the market. Here we use a simple keyword approach:
headlines containing positive words increase score, and negative words
decrease score. This lightweight approach is easy to audit and can run
quickly in production.

This module helps in two ways:
1) Avoid trading during dangerous macro-news windows.
2) Detect early news-catalyst opportunities before the full move is done.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

try:
    import requests
except ImportError:  # pragma: no cover - runtime dependency may be missing in some environments.
    requests = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - runtime dependency may be missing in some environments.
    yf = None

try:
    from AlgorithmImports import Resolution
except ImportError:  # pragma: no cover - allows local testing outside Lean runtime.
    Resolution = None


ALPHA_VANTAGE_API_KEY = "YOUR_ALPHA_VANTAGE_API_KEY"
NEWS_LOG_FILE = "news_log.json"
ET = ZoneInfo("America/New_York")

BULLISH_KEYWORDS = {
    "beat",
    "surge",
    "strong",
    "rally",
    "upgrade",
    "positive",
    "gains",
    "record",
}

BEARISH_KEYWORDS = {
    "miss",
    "drop",
    "weak",
    "cut",
    "downgrade",
    "negative",
    "loss",
    "crash",
    "fear",
}

ES_FOCUS_KEYWORDS = {
    "s&p",
    "sp500",
    "spy",
    "es",
    "e-mini",
    "fed",
    "federal reserve",
    "fomc",
    "economy",
    "cpi",
    "ppi",
    "jobs",
    "payroll",
    "gdp",
    "inflation",
}

HIGH_IMPACT_KEYWORDS = {
    "fomc",
    "federal reserve",
    "fed",
    "cpi",
    "ppi",
    "jobs report",
    "nonfarm payroll",
    "nfp",
    "gdp",
    "inflation",
    "rate decision",
}


# This function resolves a file path that works in both local and hosted runtimes.
def _resolve_path(file_name: str) -> Path:
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / file_name


# This function converts datetimes to ISO strings in a stable UTC format.
def _to_iso(value: datetime) -> str:
    utc_value = value.astimezone(timezone.utc)
    return utc_value.isoformat()


# This function parses text timestamps safely and normalizes them to UTC.
def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


@dataclass
class NoTradeWindow:
    name: str
    start: datetime
    end: datetime


class NewsEngine:
    """Fetches news in background, scores sentiment, and exposes trade guards."""

    # This function initializes thread state, logs, and runtime sentiment values.
    def __init__(
        self,
        algorithm=None,
        market_symbol=None,
        alpha_vantage_api_key: str = ALPHA_VANTAGE_API_KEY,
        fetch_interval_seconds: int = 120,
    ) -> None:
        self.algorithm = algorithm
        self.market_symbol = market_symbol
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.fetch_interval_seconds = max(15, int(fetch_interval_seconds))

        self.log_path = _resolve_path(NEWS_LOG_FILE)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.news_log: List[Dict[str, Any]] = []
        self.last_fetch_time: Optional[datetime] = None
        self.market_sentiment_score = 0.0
        self.es_sentiment_score = 0.0
        self.news_available = True

        self._load_news_log()
        self._recompute_sentiment_scores()

    # This function starts the daemon thread that fetches news every 2 minutes.
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="news-engine",
                daemon=True,
            )
            self._thread.start()
        self._log_debug("[NEWS] Background news thread started.")

    # This function stops the background fetch loop without blocking trading logic.
    def stop(self, join_timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_seconds)

    # This function runs the continuous fetch loop at the configured interval.
    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.fetch_news_once()
            except Exception as error:
                self._log_debug(f"[NEWS] Background fetch failed safely: {error}")
            self._stop_event.wait(timeout=self.fetch_interval_seconds)

    # This function fetches from all sources once and updates log/sentiment state.
    def fetch_news_once(self) -> None:
        now_utc = datetime.now(timezone.utc)
        fetched: List[Dict[str, Any]] = []
        sources_succeeded = 0

        for fetcher in (
            self._fetch_finviz_news,
            self._fetch_yfinance_news,
            self._fetch_alpha_vantage_news,
        ):
            try:
                rows = fetcher()
                sources_succeeded += 1
                fetched.extend(rows)
            except Exception as error:
                self._log_debug(f"[NEWS] Source fetch failed safely: {error}")

        with self._lock:
            self.last_fetch_time = now_utc
            self.news_available = sources_succeeded > 0

            if fetched:
                deduped = self._dedupe_headlines(fetched)
                self.news_log.extend(deduped)
                self.news_log = self.news_log[-500:]
                self._save_news_log()

            self._recompute_sentiment_scores()

        if not self.news_available:
            self._log_debug("[NEWS] All sources failed; news_available=False, trading continues normally.")

    # This function pulls market headlines from Finviz and scores each headline.
    def _fetch_finviz_news(self) -> List[Dict[str, Any]]:
        if requests is None:
            raise RuntimeError("requests library is not installed")

        response = requests.get(
            "https://finviz.com/news.ashx",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        html = response.text

        headline_matches = re.findall(
            r'class="nn-tab-link"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not headline_matches:
            # Fallback parse if Finviz HTML markup changes.
            headline_matches = re.findall(
                r"<a[^>]+href=\"[^\"]+\"[^>]*>([^<]{20,300})</a>",
                html,
                flags=re.IGNORECASE,
            )

        now_utc = datetime.now(timezone.utc)
        rows: List[Dict[str, Any]] = []
        for raw in headline_matches[:80]:
            headline = re.sub(r"\s+", " ", raw).strip()
            if len(headline) < 15:
                continue
            rows.append(self._build_news_row(now_utc, headline, "finviz"))
        return rows

    # This function pulls ES/SPY/QQQ related headlines through yfinance.
    def _fetch_yfinance_news(self) -> List[Dict[str, Any]]:
        if yf is None:
            raise RuntimeError("yfinance library is not installed")

        rows: List[Dict[str, Any]] = []
        for ticker in ("ES=F", "SPY", "QQQ"):
            news_items = yf.Ticker(ticker).news or []
            for item in news_items[:40]:
                headline = str(item.get("title") or "").strip()
                if not headline:
                    continue

                published_epoch = item.get("providerPublishTime")
                if published_epoch is None:
                    timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = datetime.fromtimestamp(int(published_epoch), tz=timezone.utc)
                rows.append(self._build_news_row(timestamp, headline, f"yfinance:{ticker}"))
        return rows

    # This function pulls economic news from Alpha Vantage free tier.
    def _fetch_alpha_vantage_news(self) -> List[Dict[str, Any]]:
        if requests is None:
            raise RuntimeError("requests library is not installed")

        if (
            not self.alpha_vantage_api_key
            or self.alpha_vantage_api_key == "YOUR_ALPHA_VANTAGE_API_KEY"
        ):
            raise RuntimeError("Alpha Vantage key placeholder is still set")

        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "NEWS_SENTIMENT",
                "topics": "economy_fiscal,economy_monetary,financial_markets",
                "limit": "100",
                "apikey": self.alpha_vantage_api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        payload = response.json()

        feed = payload.get("feed", [])
        rows: List[Dict[str, Any]] = []
        for item in feed[:100]:
            headline = str(item.get("title") or "").strip()
            if not headline:
                continue

            timestamp = self._parse_alpha_vantage_timestamp(item.get("time_published"))
            rows.append(self._build_news_row(timestamp, headline, "alpha_vantage"))
        return rows

    # This function parses Alpha Vantage timestamps and safely falls back to current UTC.
    def _parse_alpha_vantage_timestamp(self, raw: Any) -> datetime:
        if not raw:
            return datetime.now(timezone.utc)
        text = str(raw)
        try:
            # Example format: 20240130T143000
            return datetime.strptime(text, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)

    # This function builds one normalized headline record with sentiment and impact tags.
    def _build_news_row(self, timestamp: datetime, headline: str, source: str) -> Dict[str, Any]:
        sentiment = self._score_headline_sentiment(headline)
        impact = self._classify_market_impact(headline)
        return {
            "timestamp": _to_iso(timestamp),
            "headline": headline,
            "source": source,
            "sentiment_score": float(sentiment),
            "market_impact": impact,
        }

    # This function applies keyword-based sentiment scoring between -1.0 and +1.0.
    def _score_headline_sentiment(self, headline: str) -> float:
        # Sentiment analysis here is rule-based: positive keywords raise score,
        # negative keywords lower score.
        text = headline.lower()
        bullish_hits = sum(1 for keyword in BULLISH_KEYWORDS if keyword in text)
        bearish_hits = sum(1 for keyword in BEARISH_KEYWORDS if keyword in text)
        total = bullish_hits + bearish_hits
        if total == 0:
            return 0.0
        raw_score = (bullish_hits - bearish_hits) / total
        return float(max(-1.0, min(1.0, raw_score)))

    # This function classifies market impact so catalysts can be filtered by severity.
    def _classify_market_impact(self, headline: str) -> str:
        text = headline.lower()
        if any(keyword in text for keyword in HIGH_IMPACT_KEYWORDS):
            return "high"
        if any(keyword in text for keyword in ES_FOCUS_KEYWORDS):
            return "medium"
        return "low"

    # This function removes duplicate headlines to keep logs clean.
    def _dedupe_headlines(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        deduped: List[Dict[str, Any]] = []

        for row in rows:
            key = (
                str(row.get("headline", "")).strip().lower(),
                str(row.get("source", "")).strip().lower(),
                str(row.get("timestamp", ""))[:16],  # minute-level dedupe window
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
        return deduped

    # This function recomputes market and ES-focused sentiment from logged headlines.
    def _recompute_sentiment_scores(self) -> None:
        now_utc = datetime.now(timezone.utc)
        all_rows = self.news_log[-500:]
        self.market_sentiment_score = self._weighted_sentiment(all_rows, now_utc, es_only=False)
        self.es_sentiment_score = self._weighted_sentiment(all_rows, now_utc, es_only=True)

    # This function computes weighted average sentiment with 3x weight for last 30 minutes.
    def _weighted_sentiment(
        self,
        rows: List[Dict[str, Any]],
        now_utc: datetime,
        es_only: bool,
    ) -> float:
        weighted_sum = 0.0
        total_weight = 0.0

        for row in rows:
            timestamp_text = row.get("timestamp")
            headline = str(row.get("headline", ""))
            if not timestamp_text:
                continue
            try:
                timestamp = _parse_iso(str(timestamp_text))
            except ValueError:
                continue

            age_minutes = (now_utc - timestamp).total_seconds() / 60.0
            if age_minutes < 0:
                age_minutes = 0.0
            if age_minutes > 24 * 60:
                continue

            if es_only and not self._is_es_related(headline):
                continue

            sentiment = float(row.get("sentiment_score", 0.0))
            weight = 3.0 if age_minutes <= 30.0 else 1.0

            weighted_sum += sentiment * weight
            total_weight += weight

        if total_weight == 0.0:
            return 0.0
        return float(weighted_sum / total_weight)

    # This function checks if a headline is focused on ES/SPY/Fed/economy context.
    def _is_es_related(self, headline: str) -> bool:
        text = headline.lower()
        return any(keyword in text for keyword in ES_FOCUS_KEYWORDS)

    # This function returns whether current time is in a hard no-trade window and why.
    def should_avoid_trading(self, now: Optional[datetime] = None) -> Tuple[bool, str]:
        now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
        windows = self._active_no_trade_windows(now_utc)
        if not windows:
            return False, "No active no-trade windows."
        return True, windows[0].name

    # This function detects active no-trade windows from hardcoded event timing rules.
    def _active_no_trade_windows(self, now_utc: datetime) -> List[NoTradeWindow]:
        now_et = now_utc.astimezone(ET)
        windows = self._build_daily_windows(now_et.date())
        return [window for window in windows if window.start <= now_et <= window.end]

    # This function builds hardcoded daily event windows for major risk events.
    def _build_daily_windows(self, day) -> List[NoTradeWindow]:
        windows: List[NoTradeWindow] = []

        # Market open/close windows (weekdays only).
        if day.weekday() < 5:
            open_dt = datetime.combine(day, time(9, 30), tzinfo=ET)
            close_dt = datetime.combine(day, time(16, 0), tzinfo=ET)
            windows.append(
                NoTradeWindow(
                    name="No trade: first 5 minutes after market open",
                    start=open_dt,
                    end=open_dt + timedelta(minutes=5),
                )
            )
            windows.append(
                NoTradeWindow(
                    name="No trade: last 10 minutes before market close",
                    start=close_dt - timedelta(minutes=10),
                    end=close_dt,
                )
            )

        # Hardcoded macro-event windows.
        events = [
            ("FOMC decision window", time(14, 0), 30, 15),
            ("Fed speech window", time(14, 30), 30, 15),
            ("CPI/PPI/Jobs release window", time(8, 30), 15, 10),
            ("GDP release window", time(8, 30), 15, 10),
        ]
        for name, event_time, minutes_before, minutes_after in events:
            center = datetime.combine(day, event_time, tzinfo=ET)
            windows.append(
                NoTradeWindow(
                    name=name,
                    start=center - timedelta(minutes=minutes_before),
                    end=center + timedelta(minutes=minutes_after),
                )
            )
        return windows

    # This function returns the next no-trade window starting within a lookahead horizon.
    def _upcoming_no_trade_window(
        self,
        now_utc: datetime,
        lookahead_minutes: int,
    ) -> Optional[NoTradeWindow]:
        now_et = now_utc.astimezone(ET)
        windows = self._build_daily_windows(now_et.date()) + self._build_daily_windows(
            (now_et + timedelta(days=1)).date()
        )
        horizon = now_et + timedelta(minutes=lookahead_minutes)

        upcoming = [
            window
            for window in windows
            if now_et <= window.start <= horizon or (window.start <= now_et <= window.end)
        ]
        if not upcoming:
            return None
        upcoming.sort(key=lambda item: item.start)
        return upcoming[0]

    # This function finds an early catalyst opportunity if strict conditions are met.
    def find_opportunity(self, now: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        now_utc = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)

        avoid, reason = self.should_avoid_trading(now_utc)
        if avoid:
            return None

        upcoming = self._upcoming_no_trade_window(now_utc, lookahead_minutes=20)
        if upcoming is not None:
            return None

        with self._lock:
            rows = list(self.news_log[-200:])
            sentiment = float(self.market_sentiment_score)
            available = bool(self.news_available)

        if not available:
            return None

        recent_high_impact = [
            row
            for row in rows
            if row.get("market_impact") == "high"
            and (_parse_iso(str(row["timestamp"])) >= now_utc - timedelta(minutes=10))
        ]
        if not recent_high_impact:
            return None

        latest = max(recent_high_impact, key=lambda row: str(row.get("timestamp", "")))
        latest_sentiment = float(latest.get("sentiment_score", 0.0))

        # Required bullish catalyst rule from spec.
        if latest_sentiment <= 0.0 or sentiment <= 0.5:
            return None

        market_move = self._estimate_market_move_10m()
        if market_move is None:
            return None
        if abs(market_move) > 0.003:
            return None

        confidence = float(min(1.0, 0.60 + (sentiment - 0.5)))
        return {
            "direction": "long",
            "catalyst": str(latest.get("headline", "High-impact bullish catalyst")),
            "confidence_score": round(confidence, 4),
            "market_move_10m": round(float(market_move), 6),
            "no_trade_reason": reason,
        }

    # This function estimates market move over the last 10 minutes for early-entry filtering.
    def _estimate_market_move_10m(self) -> Optional[float]:
        if self.algorithm is None or self.market_symbol is None:
            return None

        try:
            if Resolution is None:
                return None
            history = self.algorithm.History(self.market_symbol, timedelta(minutes=12), Resolution.Minute)
        except Exception as error:
            self._log_debug(f"[NEWS] Market move estimation failed: {error}")
            return None

        frame = self._normalize_history_frame(history, self.market_symbol)
        if frame is None or len(frame) < 3:
            return None

        start = float(frame["close"].iloc[0])
        end = float(frame["close"].iloc[-1])
        if start == 0.0:
            return None
        return (end - start) / start

    # This function normalizes history output into a close-price DataFrame.
    def _normalize_history_frame(self, history, symbol) -> Optional[Any]:
        if history is None or getattr(history, "empty", True):
            return None

        frame = history
        if "symbol" in getattr(frame.index, "names", []):
            try:
                frame = frame.xs(symbol, level="symbol")
            except Exception:
                return None

        renamed = frame.rename(columns={"Close": "close"})
        lower_map = {str(column).lower(): column for column in renamed.columns}
        if "close" not in lower_map:
            return None
        column = lower_map["close"]
        normalized = renamed[[column]].copy()
        normalized.columns = ["close"]
        return normalized.dropna()

    # This function returns current news engine state for monitoring dashboards.
    def get_news_status(self) -> Dict[str, Any]:
        now_utc = datetime.now(timezone.utc)
        active_windows = [window.name for window in self._active_no_trade_windows(now_utc)]
        live_opportunity = self.find_opportunity(now_utc)

        with self._lock:
            last_fetch_text = self.last_fetch_time.isoformat() if self.last_fetch_time else None
            market_sentiment = float(self.market_sentiment_score)
            es_sentiment = float(self.es_sentiment_score)
            available = bool(self.news_available)

        return {
            "last_fetch_time": last_fetch_text,
            "market_sentiment_score": round(market_sentiment, 6),
            "es_sentiment_score": round(es_sentiment, 6),
            "news_available": available,
            "active_no_trade_windows": active_windows,
            "live_opportunity": live_opportunity,
        }

    # This function loads prior headline logs from disk on startup.
    def _load_news_log(self) -> None:
        try:
            if self.log_path.exists():
                with self.log_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        self.news_log = loaded[-500:]
        except (OSError, json.JSONDecodeError):
            self.news_log = []

    # This function saves the latest headline log, capped to 500 rows.
    def _save_news_log(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("w", encoding="utf-8") as handle:
                json.dump(self.news_log[-500:], handle, indent=2)
        except OSError as error:
            self._log_debug(f"[NEWS] Could not write news_log.json: {error}")

    # This function logs debug output through algorithm logger when available.
    def _log_debug(self, message: str) -> None:
        if self.algorithm is not None and hasattr(self.algorithm, "Debug"):
            try:
                self.algorithm.Debug(message)
                return
            except Exception:
                pass
        # Safe fallback for local/offline testing.
        print(message)
