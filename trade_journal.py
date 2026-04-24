"""Trade journaling and learning feedback module for live improvement.

A feedback loop means the bot records what happened, studies the results,
and then uses what it learned to improve future decisions. Learning from
real executed trades is often more valuable than only historical backtests
because live fills, slippage, and real market behavior reveal what actually
works under real conditions.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# This function returns a file path that works in local and hosted runtimes.
def _resolve_path(file_name: str) -> Path:
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / file_name


# This function converts date-time values into stable ISO text.
def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    return str(value)


# This dataclass stores derived rolling performance values.
@dataclass
class RollingStats:
    win_rate: float
    average_winner_size: float
    average_loser_size: float
    profit_factor: float


class TradeJournal:
    """Records trades, learns from outcomes, and exposes performance state."""

    # This function sets up in-memory state and loads existing files from disk.
    def __init__(
        self,
        trades_path: Optional[Path] = None,
        insights_path: Optional[Path] = None,
        learning_cycle_size: int = 20,
    ) -> None:
        self.trades_path = Path(trades_path) if trades_path else _resolve_path("trades_history.json")
        self.insights_path = (
            Path(insights_path) if insights_path else _resolve_path("learned_insights.json")
        )
        self.learning_cycle_size = max(1, int(learning_cycle_size))

        self._lock = threading.Lock()
        self.trades: List[Dict[str, Any]] = []
        self.learned_insights: List[Dict[str, Any]] = []
        self.open_trades: Dict[str, Dict[str, Any]] = {}

        self.completed_since_learning = 0
        self.confidence_low = False
        self._rolling_stats = RollingStats(0.0, 0.0, 0.0, 0.0)
        self._pending_training_feedback: List[Dict[str, Any]] = []

        self._load_trades()
        self._load_insights()
        self._rebuild_open_trades()
        self._refresh_rolling_stats()

    # This function records a newly opened trade with full entry context.
    def record_trade_open(
        self,
        entry_time: Any,
        instrument: str,
        direction: str,
        entry_price: float,
        contract_size: int,
        rsi_value: float,
        ma_fast_value: float,
        ma_slow_value: float,
        momentum_value: float,
        volatility_value: float,
        volume_value: float,
        news_sentiment_score: float,
        ai_confidence_score: float,
        reasoning_text: str,
        stop_loss_level: float,
        take_profit_level: float,
        active_parameter_set: Dict[str, Any],
        trade_id: Optional[str] = None,
    ) -> str:
        direction_lower = direction.lower()
        if direction_lower not in ("long", "short"):
            raise ValueError("direction must be either 'long' or 'short'")

        resolved_trade_id = str(trade_id) if trade_id else str(uuid.uuid4())
        trade = {
            "trade_id": resolved_trade_id,
            "status": "open",
            "entry_time": _to_iso(entry_time),
            "instrument": instrument,
            "direction": direction_lower,
            "entry_price": float(entry_price),
            "contract_size": int(contract_size),
            "entry_conditions": {
                "rsi": float(rsi_value),
                "ma_fast": float(ma_fast_value),
                "ma_slow": float(ma_slow_value),
                "momentum": float(momentum_value),
                "volatility": float(volatility_value),
                "volume": float(volume_value),
                "news_sentiment_score": float(news_sentiment_score),
                "ai_confidence_score": float(ai_confidence_score),
            },
            "reasoning_text": str(reasoning_text),
            "stop_loss_level": float(stop_loss_level),
            "take_profit_level": float(take_profit_level),
            "active_parameter_set": dict(active_parameter_set),
            "exit_time": None,
            "exit_price": None,
            "exit_reason": None,
            "pnl_points": None,
            "pnl_dollars": None,
            "trade_duration_minutes": None,
            "winner": None,
        }

        with self._lock:
            self.trades.append(trade)
            self.open_trades[resolved_trade_id] = trade
            self._save_trades()
        return resolved_trade_id

    # This function records the close details and final outcome for a trade.
    def record_trade_close(
        self,
        trade_id: str,
        exit_time: Any,
        exit_price: float,
        exit_reason: str,
        point_value: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            trade = self.open_trades.get(trade_id)
            if trade is None:
                raise KeyError(f"trade_id not found among open trades: {trade_id}")

            instrument = str(trade["instrument"]).upper()
            resolved_point_value = (
                float(point_value) if point_value is not None else self._default_point_value(instrument)
            )

            direction_mult = 1.0 if trade["direction"] == "long" else -1.0
            entry_price = float(trade["entry_price"])
            final_exit_price = float(exit_price)
            contract_size = int(trade["contract_size"])

            pnl_points = (final_exit_price - entry_price) * direction_mult
            pnl_dollars = pnl_points * resolved_point_value * contract_size

            entry_dt = self._parse_iso_datetime(trade["entry_time"])
            exit_dt = self._parse_iso_datetime(_to_iso(exit_time))
            trade_duration_minutes = (exit_dt - entry_dt).total_seconds() / 60.0

            trade["status"] = "closed"
            trade["exit_time"] = exit_dt.isoformat()
            trade["exit_price"] = final_exit_price
            trade["exit_reason"] = str(exit_reason)
            trade["pnl_points"] = float(pnl_points)
            trade["pnl_dollars"] = float(pnl_dollars)
            trade["trade_duration_minutes"] = float(max(0.0, trade_duration_minutes))
            trade["winner"] = bool(pnl_dollars > 0.0)

            self.open_trades.pop(trade_id, None)
            self.completed_since_learning += 1

            self._save_trades()
            self._refresh_rolling_stats()

            if self.completed_since_learning >= self.learning_cycle_size:
                self._run_learning_cycle()
                self.completed_since_learning = 0

            return dict(trade)

    # This function returns a 50% size multiplier during weak performance periods.
    def get_position_size_multiplier(self) -> float:
        return 0.5 if self.confidence_low else 1.0

    # This function returns weighted training examples for the next model retrain.
    def get_pending_training_feedback(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(item) for item in self._pending_training_feedback]

    # This function returns and clears pending weighted feedback after consumption.
    def consume_pending_training_feedback(self) -> List[Dict[str, Any]]:
        with self._lock:
            payload = [dict(item) for item in self._pending_training_feedback]
            self._pending_training_feedback = []
            return payload

    # This function returns rolling last-20 performance stats and confidence state.
    def get_rolling_stats(self) -> Dict[str, float]:
        with self._lock:
            return {
                "win_rate_last_20": round(float(self._rolling_stats.win_rate), 6),
                "average_winner_size_last_20": round(
                    float(self._rolling_stats.average_winner_size), 2
                ),
                "average_loser_size_last_20": round(
                    float(self._rolling_stats.average_loser_size), 2
                ),
                "profit_factor_last_20": round(float(self._rolling_stats.profit_factor), 6),
                "confidence_low": bool(self.confidence_low),
            }

    # This function summarizes journal performance and recent learned insights.
    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            closed = [trade for trade in self.trades if trade.get("status") == "closed"]
            total_trades = len(closed)
            wins = [trade for trade in closed if trade.get("winner")]
            win_rate = (len(wins) / total_trades) if total_trades else 0.0

            pnl_values = [float(trade.get("pnl_dollars") or 0.0) for trade in closed]
            total_pnl = float(sum(pnl_values))

            best_trade = max(closed, key=lambda item: float(item.get("pnl_dollars") or 0.0), default=None)
            worst_trade = min(closed, key=lambda item: float(item.get("pnl_dollars") or 0.0), default=None)

            top_3_insights = [
                insight.get("insight", "")
                for insight in sorted(
                    self.learned_insights,
                    key=lambda row: row.get("timestamp", ""),
                    reverse=True,
                )[:3]
            ]

            return {
                "total_trades": total_trades,
                "win_rate": round(float(win_rate), 6),
                "total_pnl": round(total_pnl, 2),
                "best_trade": best_trade,
                "worst_trade": worst_trade,
                "confidence_low": bool(self.confidence_low),
                "top_3_learned_insights": top_3_insights,
            }

    # This function loads historic trade records from trades_history.json.
    def _load_trades(self) -> None:
        try:
            if self.trades_path.exists():
                with self.trades_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        self.trades = loaded
        except (OSError, json.JSONDecodeError):
            self.trades = []

    # This function loads saved learning insights from learned_insights.json.
    def _load_insights(self) -> None:
        try:
            if self.insights_path.exists():
                with self.insights_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        self.learned_insights = loaded
        except (OSError, json.JSONDecodeError):
            self.learned_insights = []

    # This function rebuilds the open-trade index from loaded trade history.
    def _rebuild_open_trades(self) -> None:
        self.open_trades = {}
        for trade in self.trades:
            if trade.get("status") == "open":
                trade_id = str(trade.get("trade_id", ""))
                if trade_id:
                    self.open_trades[trade_id] = trade

        closed_count = len([trade for trade in self.trades if trade.get("status") == "closed"])
        self.completed_since_learning = closed_count % self.learning_cycle_size

    # This function updates rolling stats and toggles the confidence_low flag.
    def _refresh_rolling_stats(self) -> None:
        closed = [trade for trade in self.trades if trade.get("status") == "closed"]
        recent = closed[-20:]

        if not recent:
            self._rolling_stats = RollingStats(0.0, 0.0, 0.0, 0.0)
            self.confidence_low = False
            return

        pnl_values = [float(trade.get("pnl_dollars") or 0.0) for trade in recent]
        winners = [value for value in pnl_values if value > 0.0]
        losers = [value for value in pnl_values if value < 0.0]

        win_rate = len(winners) / len(recent)
        avg_winner = (sum(winners) / len(winners)) if winners else 0.0
        avg_loser = (abs(sum(losers)) / len(losers)) if losers else 0.0
        profit_factor = (sum(winners) / abs(sum(losers))) if losers else (sum(winners) if winners else 0.0)

        self._rolling_stats = RollingStats(
            win_rate=float(win_rate),
            average_winner_size=float(avg_winner),
            average_loser_size=float(avg_loser),
            profit_factor=float(profit_factor),
        )

        # We only enforce the low-confidence rule after at least 20 finished trades.
        if len(recent) == 20 and win_rate < 0.40:
            self.confidence_low = True
        elif len(recent) == 20 and win_rate >= 0.40:
            self.confidence_low = False

    # This function runs one learning cycle that extracts and saves trade patterns.
    def _run_learning_cycle(self) -> None:
        closed = [trade for trade in self.trades if trade.get("status") == "closed"]
        if len(closed) < self.learning_cycle_size:
            return

        sample = closed[-self.learning_cycle_size :]
        winners = [trade for trade in sample if trade.get("winner")]
        losers = [trade for trade in sample if not trade.get("winner")]
        if not winners or not losers:
            return

        patterns = self._discover_patterns(winners, losers)
        insight_text = self._build_insight_text(patterns)
        insight = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sample_size": len(sample),
            "winner_count": len(winners),
            "loser_count": len(losers),
            "insight": insight_text,
            "patterns": patterns,
        }

        self.learned_insights.append(insight)
        self._save_insights()

        self._pending_training_feedback = self._build_weighted_feedback(winners, losers)

    # This function identifies winner-heavy versus loser-light condition patterns.
    def _discover_patterns(
        self,
        winners: List[Dict[str, Any]],
        losers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        winner_tags = [self._condition_tags(trade) for trade in winners]
        loser_tags = [self._condition_tags(trade) for trade in losers]

        all_keys = sorted({key for tags in winner_tags + loser_tags for key in tags})
        found: List[Dict[str, Any]] = []
        for key in all_keys:
            winner_rate = sum(1 for tags in winner_tags if key in tags) / max(1, len(winner_tags))
            loser_rate = sum(1 for tags in loser_tags if key in tags) / max(1, len(loser_tags))

            if winner_rate >= 0.70 and loser_rate < 0.30:
                found.append(
                    {
                        "pattern_key": key,
                        "winner_frequency": round(winner_rate, 4),
                        "loser_frequency": round(loser_rate, 4),
                        "edge": round(winner_rate - loser_rate, 4),
                    }
                )

        found.sort(key=lambda row: row["edge"], reverse=True)
        return found

    # This function converts one trade's entry conditions into simple pattern tags.
    def _condition_tags(self, trade: Dict[str, Any]) -> List[str]:
        cond = trade.get("entry_conditions", {})
        rsi = float(cond.get("rsi", 50.0))
        ma_fast = float(cond.get("ma_fast", 0.0))
        ma_slow = float(cond.get("ma_slow", 0.0))
        momentum = float(cond.get("momentum", 0.0))
        volatility = float(cond.get("volatility", 0.0))
        volume = float(cond.get("volume", 0.0))
        sentiment = float(cond.get("news_sentiment_score", 0.0))
        confidence = float(cond.get("ai_confidence_score", 0.0))

        tags: List[str] = []
        if rsi <= 35:
            tags.append("rsi_low")
        elif rsi >= 65:
            tags.append("rsi_high")
        else:
            tags.append("rsi_mid")

        if ma_fast > ma_slow:
            tags.append("ma_trend_up")
        else:
            tags.append("ma_trend_down")

        tags.append("momentum_positive" if momentum > 0 else "momentum_negative")

        if volatility < 0.004:
            tags.append("volatility_low")
        elif volatility < 0.012:
            tags.append("volatility_medium")
        else:
            tags.append("volatility_high")

        tags.append("volume_above_ref" if volume >= 1.0 else "volume_below_ref")
        if sentiment > 0.2:
            tags.append("sentiment_positive")
        elif sentiment < -0.2:
            tags.append("sentiment_negative")
        else:
            tags.append("sentiment_neutral")

        tags.append("confidence_high" if confidence >= 0.70 else "confidence_moderate")
        tags.append(f"direction_{trade.get('direction', 'unknown')}")
        return tags

    # This function converts machine-readable pattern keys into plain English insight text.
    def _build_insight_text(self, patterns: List[Dict[str, Any]]) -> str:
        if not patterns:
            return (
                "No strong winner-specific pattern reached the 70%/30% threshold this cycle, "
                "so the bot should keep collecting more trades."
            )

        key_to_text = {
            "rsi_low": "RSI was in a lower zone",
            "rsi_high": "RSI was in a higher zone",
            "rsi_mid": "RSI was in a neutral zone",
            "ma_trend_up": "fast moving average was above slow moving average",
            "ma_trend_down": "fast moving average was below slow moving average",
            "momentum_positive": "momentum was positive",
            "momentum_negative": "momentum was negative",
            "volatility_low": "volatility was low",
            "volatility_medium": "volatility was medium",
            "volatility_high": "volatility was high",
            "volume_above_ref": "volume was above the reference level",
            "volume_below_ref": "volume was below the reference level",
            "sentiment_positive": "news sentiment was positive",
            "sentiment_negative": "news sentiment was negative",
            "sentiment_neutral": "news sentiment was neutral",
            "confidence_high": "AI confidence was high",
            "confidence_moderate": "AI confidence was moderate",
            "direction_long": "trade direction was long",
            "direction_short": "trade direction was short",
        }

        top = patterns[:3]
        descriptions = [key_to_text.get(row["pattern_key"], row["pattern_key"]) for row in top]
        return (
            "Winners in the latest learning cycle were most often seen when "
            + ", ".join(descriptions)
            + "."
        )

    # This function prepares weighted trade-condition samples for the next AI retrain.
    def _build_weighted_feedback(
        self,
        winners: List[Dict[str, Any]],
        losers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        weighted: List[Dict[str, Any]] = []
        for trade in winners:
            weighted.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "label": 1,
                    "weight": 2.0,
                    "conditions": dict(trade.get("entry_conditions", {})),
                }
            )
        for trade in losers:
            weighted.append(
                {
                    "trade_id": trade.get("trade_id"),
                    "label": 0,
                    "weight": 1.0,
                    "conditions": dict(trade.get("entry_conditions", {})),
                }
            )
        return weighted

    # This function writes full trade history to disk without deleting prior records.
    def _save_trades(self) -> None:
        try:
            self.trades_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trades_path.open("w", encoding="utf-8") as handle:
                json.dump(self.trades, handle, indent=2)
        except OSError:
            # File-write issues should not stop live trading logic.
            return

    # This function writes learned insights to disk for future review.
    def _save_insights(self) -> None:
        try:
            self.insights_path.parent.mkdir(parents=True, exist_ok=True)
            with self.insights_path.open("w", encoding="utf-8") as handle:
                json.dump(self.learned_insights, handle, indent=2)
        except OSError:
            # File-write issues should not stop live trading logic.
            return

    # This function resolves default point value by instrument when one is not provided.
    def _default_point_value(self, instrument: str) -> float:
        mapping = {
            "ES": 50.0,
            "NQ": 20.0,
            "MES": 5.0,
            "MNQ": 2.0,
        }
        return float(mapping.get(instrument.upper(), 1.0))

    # This function parses stored ISO timestamps safely and always returns UTC datetime.
    def _parse_iso_datetime(self, value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
