"""Plain-English trade reasoning engine for human-readable decision logs.

This module turns bot decisions into clear explanations that people can read
without trading knowledge. It records why trades were opened or closed, what
signals mattered most, and what lessons were learned each day.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# This function resolves a runtime-safe file path for logs and JSON storage.
def _resolve_path(file_name: str) -> Path:
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / file_name


# This function converts datetimes to consistent ISO-8601 UTC text.
def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


class TradeReasoningEngine:
    """Generates and saves plain-English reasoning for entries, exits, and day summaries."""

    # This function initializes storage locations and loads prior reasoning entries.
    def __init__(
        self,
        reasons_json_path: Optional[Path] = None,
        reasons_log_path: Optional[Path] = None,
    ) -> None:
        self.reasons_json_path = (
            Path(reasons_json_path) if reasons_json_path else _resolve_path("trade_reasons.json")
        )
        self.reasons_log_path = (
            Path(reasons_log_path) if reasons_log_path else _resolve_path("trade_reasons.log")
        )
        self._lock = threading.Lock()
        self.reasons: List[Dict[str, Any]] = []
        self._load_reasons()

    # This function creates a full plain-English entry explanation and saves it.
    def generate_entry_reason(
        self,
        trade_id: str,
        timestamp: Any,
        instrument: str,
        direction: str,
        technicals: Dict[str, Any],
        ai_context: Dict[str, Any],
        news_context: Dict[str, Any],
        risk_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        rsi_value = float(technicals.get("rsi", 50.0))
        ma_fast = float(technicals.get("ma_fast", 0.0))
        ma_slow = float(technicals.get("ma_slow", 0.0))
        momentum = float(technicals.get("momentum", 0.0))
        volatility = float(technicals.get("volatility", 0.0))
        volume_relative = float(technicals.get("volume_relative", 1.0))

        signal_lines, mixed_lines = self._build_signal_summary_lines(
            rsi_value=rsi_value,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            momentum=momentum,
            volatility=volatility,
            volume_relative=volume_relative,
        )
        technical_bias = self._compute_technical_bias(
            rsi_value=rsi_value,
            ma_fast=ma_fast,
            ma_slow=ma_slow,
            momentum=momentum,
            direction=direction,
        )

        model_confidence = float(ai_context.get("confidence", 0.0))
        top_features = list(ai_context.get("top_features", []))[:3]
        while len(top_features) < 3:
            top_features.append("feature not available")

        historical_match = str(ai_context.get("historical_similarity_text", "No similarity data provided."))
        in_pattern = bool(ai_context.get("in_pattern", False))
        pattern_label = "in-pattern" if in_pattern else "exploratory"

        market_sentiment = float(news_context.get("market_sentiment_score", 0.0))
        relevant_headlines = list(news_context.get("relevant_headlines", []))[:3]
        if not relevant_headlines:
            relevant_headlines = ["No major headline was flagged as directly relevant."]
        supports_signal = bool(news_context.get("supports_signal", False))
        upcoming_events = list(news_context.get("upcoming_events", []))
        if not upcoming_events:
            upcoming_events = ["No high-impact event is scheduled in the immediate window."]

        entry_price = float(risk_context.get("entry_price", 0.0))
        stop_loss = float(risk_context.get("stop_loss", 0.0))
        take_profit = float(risk_context.get("take_profit", 0.0))
        point_value = float(risk_context.get("point_value", 1.0))
        quantity = int(risk_context.get("quantity", 1))
        daily_pnl = float(risk_context.get("daily_pnl", 0.0))
        daily_loss_limit = float(risk_context.get("daily_loss_limit", 0.0))
        trailing_drawdown = float(risk_context.get("trailing_drawdown", 0.0))
        trailing_limit = float(risk_context.get("trailing_drawdown_limit", 0.0))

        risk_points = abs(entry_price - stop_loss)
        reward_points = abs(take_profit - entry_price)
        risk_dollars = risk_points * point_value * quantity
        reward_dollars = reward_points * point_value * quantity
        rr_ratio = (reward_points / risk_points) if risk_points > 0 else 0.0

        daily_room_left = daily_loss_limit - daily_pnl
        drawdown_room_left = trailing_limit - trailing_drawdown

        verdict, verdict_sentence = self._build_confidence_verdict(
            model_confidence=model_confidence,
            technical_bias=technical_bias,
            supports_signal=supports_signal,
            pattern_label=pattern_label,
            instrument=instrument,
            direction=direction,
        )

        structured = {
            "SIGNAL SUMMARY": {
                "signals_fired": signal_lines,
                "mixed_or_borderline": mixed_lines,
                "overall_technical_bias": technical_bias,
            },
            "AI MODEL REASONING": {
                "model_confidence_percent": round(model_confidence * 100.0, 2),
                "top_3_features": top_features,
                "historical_setup_comparison": historical_match,
                "trade_type": pattern_label,
            },
            "NEWS CONTEXT": {
                "current_sentiment_score": round(market_sentiment, 4),
                "sentiment_meaning": self._describe_sentiment(market_sentiment),
                "relevant_headlines": relevant_headlines,
                "news_alignment": (
                    "News supports the trade direction." if supports_signal else "News does not clearly support this direction."
                ),
                "upcoming_news_events": upcoming_events,
            },
            "RISK ASSESSMENT": {
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_points": round(risk_points, 4),
                "reward_points": round(reward_points, 4),
                "risk_dollars": round(risk_dollars, 2),
                "reward_dollars": round(reward_dollars, 2),
                "risk_reward_ratio": round(rr_ratio, 4),
                "current_daily_pnl": round(daily_pnl, 2),
                "room_before_daily_limit": round(daily_room_left, 2),
                "current_trailing_drawdown": round(trailing_drawdown, 2),
                "room_before_trailing_limit": round(drawdown_room_left, 2),
            },
            "CONFIDENCE VERDICT": {
                "level": verdict,
                "summary": verdict_sentence,
            },
        }

        entry = {
            "trade_id": trade_id,
            "type": "entry",
            "timestamp": _to_iso(timestamp),
            "instrument": instrument,
            "direction": direction,
            "reasoning": structured,
        }
        self._store_reason(entry)
        return entry

    # This function creates a plain-English explanation for why a trade exited.
    def generate_exit_reason(
        self,
        trade_id: str,
        timestamp: Any,
        instrument: str,
        direction: str,
        exit_type: str,
        market_during_trade: Dict[str, Any],
        thesis_matched: bool,
        lesson_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        price_move_points = float(market_during_trade.get("price_move_points", 0.0))
        volatility_phase = str(market_during_trade.get("volatility_phase", "unknown"))
        momentum_shift = str(market_during_trade.get("momentum_shift", "unknown"))
        pnl_points = float(market_during_trade.get("pnl_points", 0.0))
        pnl_dollars = float(market_during_trade.get("pnl_dollars", 0.0))

        outcome_line = (
            "The outcome matched the original thesis."
            if thesis_matched
            else "The outcome did not match the original thesis."
        )
        lesson = lesson_hint or self._default_lesson_from_exit(
            exit_type=exit_type,
            thesis_matched=thesis_matched,
            pnl_dollars=pnl_dollars,
            direction=direction,
        )

        structured = {
            "EXIT TYPE": str(exit_type),
            "MARKET ACTION DURING TRADE": {
                "price_move_points": round(price_move_points, 4),
                "volatility_phase": volatility_phase,
                "momentum_shift": momentum_shift,
                "pnl_points": round(pnl_points, 4),
                "pnl_dollars": round(pnl_dollars, 2),
            },
            "THESIS CHECK": outcome_line,
            "LESSON": lesson,
        }

        entry = {
            "trade_id": trade_id,
            "type": "exit",
            "timestamp": _to_iso(timestamp),
            "instrument": instrument,
            "direction": direction,
            "reasoning": structured,
        }
        self._store_reason(entry)
        return entry

    # This function generates a full end-of-day summary that references trade reasoning and lessons.
    def generate_daily_summary(
        self,
        day: Any,
        trades_for_day: List[Dict[str, Any]],
        learned_insights: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        winners = [trade for trade in trades_for_day if bool(trade.get("winner"))]
        losers = [trade for trade in trades_for_day if trade.get("winner") is False]

        best_trade = max(
            trades_for_day,
            key=lambda item: float(item.get("pnl_dollars") or 0.0),
            default=None,
        )
        worst_trade = min(
            trades_for_day,
            key=lambda item: float(item.get("pnl_dollars") or 0.0),
            default=None,
        )

        dominated_conditions = self._infer_dominant_conditions(trades_for_day)
        top_insights = learned_insights[-3:]
        focus = self._recommend_tomorrow_focus(trades_for_day, learned_insights)

        summary = {
            "trade_id": f"daily-summary-{_to_iso(datetime.now(timezone.utc))}",
            "type": "daily_summary",
            "timestamp": _to_iso(datetime.now(timezone.utc)),
            "day": _to_iso(day),
            "reasoning": {
                "TOTALS": {
                    "total_trades": len(trades_for_day),
                    "winners": len(winners),
                    "losers": len(losers),
                },
                "BEST_TRADE": best_trade,
                "WORST_TRADE": worst_trade,
                "DOMINANT_MARKET_CONDITIONS": dominated_conditions,
                "AI_LEARNED_TODAY": top_insights,
                "RECOMMENDED_FOCUS_TOMORROW": focus,
            },
        }
        self._store_reason(summary)
        return summary

    # This function extracts a one-line plain-English summary from an entry reasoning record.
    def extract_entry_summary_sentence(self, entry_reason_record: Dict[str, Any]) -> str:
        reasoning = entry_reason_record.get("reasoning", {})
        verdict_section = reasoning.get("CONFIDENCE VERDICT", {})
        sentence = str(verdict_section.get("summary") or "").strip()
        if sentence:
            return sentence
        return "The bot did not receive enough context to generate a detailed one-line summary."

    # This function builds human-friendly lines explaining which technical signals fired.
    def _build_signal_summary_lines(
        self,
        rsi_value: float,
        ma_fast: float,
        ma_slow: float,
        momentum: float,
        volatility: float,
        volume_relative: float,
    ) -> (List[str], List[str]):
        fired: List[str] = []
        mixed: List[str] = []

        if rsi_value <= 30.0:
            fired.append(f"RSI hit {rsi_value:.2f} — oversold territory, historically a long signal.")
        elif rsi_value >= 70.0:
            fired.append(f"RSI hit {rsi_value:.2f} — overbought territory, historically a short signal.")
        else:
            mixed.append(f"RSI is {rsi_value:.2f}, which is neutral and not an extreme signal.")

        if ma_fast > ma_slow:
            fired.append(
                f"Fast moving average ({ma_fast:.2f}) is above slow moving average ({ma_slow:.2f}) — trend bias is bullish."
            )
        elif ma_fast < ma_slow:
            fired.append(
                f"Fast moving average ({ma_fast:.2f}) is below slow moving average ({ma_slow:.2f}) — trend bias is bearish."
            )
        else:
            mixed.append(
                f"Fast and slow moving averages are almost equal ({ma_fast:.2f} vs {ma_slow:.2f}), giving a mixed trend signal."
            )

        if abs(momentum) < 0.0005:
            mixed.append(f"Momentum is near flat at {momentum:.5f}, so price acceleration is weak.")
        elif momentum > 0:
            fired.append(f"Momentum is positive ({momentum:.5f}), confirming upward pressure.")
        else:
            fired.append(f"Momentum is negative ({momentum:.5f}), confirming downward pressure.")

        if volatility > 0.012:
            mixed.append(f"Volatility is elevated ({volatility:.5f}), which increases uncertainty.")
        else:
            fired.append(f"Volatility is controlled ({volatility:.5f}), making execution risk lower.")

        if volume_relative >= 1.2:
            fired.append(
                f"Relative volume is strong ({volume_relative:.2f}x normal), so signal participation looks real."
            )
        elif volume_relative < 0.8:
            mixed.append(
                f"Relative volume is light ({volume_relative:.2f}x normal), so conviction may be weaker."
            )
        else:
            mixed.append(f"Relative volume is average ({volume_relative:.2f}x normal), neither strong nor weak.")

        return fired, mixed

    # This function converts raw technical values into a single bias label.
    def _compute_technical_bias(
        self,
        rsi_value: float,
        ma_fast: float,
        ma_slow: float,
        momentum: float,
        direction: str,
    ) -> str:
        score = 0
        if rsi_value <= 30:
            score += 1
        elif rsi_value >= 70:
            score -= 1

        if ma_fast > ma_slow:
            score += 1
        elif ma_fast < ma_slow:
            score -= 1

        if momentum > 0:
            score += 1
        elif momentum < 0:
            score -= 1

        if score >= 2:
            return "strong long"
        if score == 1:
            return "weak long"
        if score == 0:
            return "neutral"
        if score == -1:
            return "weak short"
        return "strong short"

    # This function translates sentiment score into plain English text.
    def _describe_sentiment(self, sentiment: float) -> str:
        if sentiment >= 0.5:
            return "News tone is strongly bullish for risk assets."
        if sentiment >= 0.15:
            return "News tone is mildly bullish."
        if sentiment <= -0.5:
            return "News tone is strongly bearish and risk-off."
        if sentiment <= -0.15:
            return "News tone is mildly bearish."
        return "News tone is balanced/neutral."

    # This function creates a final confidence verdict and one-sentence explanation.
    def _build_confidence_verdict(
        self,
        model_confidence: float,
        technical_bias: str,
        supports_signal: bool,
        pattern_label: str,
        instrument: str,
        direction: str,
    ) -> (str, str):
        if model_confidence >= 0.75 and supports_signal and "strong" in technical_bias:
            level = "HIGH"
        elif model_confidence >= 0.60:
            level = "MEDIUM"
        else:
            level = "LOW"

        summary = (
            f"The bot is taking this {direction.lower()} {instrument} trade because model confidence is "
            f"{model_confidence * 100:.1f}%, technical bias is {technical_bias}, and the setup is "
            f"classified as {pattern_label}."
        )
        return level, summary

    # This function creates a fallback lesson sentence when no custom lesson is provided.
    def _default_lesson_from_exit(
        self,
        exit_type: str,
        thesis_matched: bool,
        pnl_dollars: float,
        direction: str,
    ) -> str:
        if thesis_matched and pnl_dollars > 0:
            return f"The {direction.lower()} thesis worked as expected; reinforce this setup when conditions repeat."
        if exit_type.lower() in ("stop hit", "risk rule triggered"):
            return "Risk controls did their job; reduce exposure or wait for cleaner confirmation next time."
        if exit_type.lower() in ("target hit",):
            return "Profit target execution was disciplined; keep taking predefined exits."
        return "Review this setup for mixed signals and tighten entry filters on similar conditions."

    # This function infers dominant market conditions from day trade metadata.
    def _infer_dominant_conditions(self, trades_for_day: List[Dict[str, Any]]) -> str:
        if not trades_for_day:
            return "No trades were taken, so no dominant condition can be inferred."

        tags = []
        for trade in trades_for_day:
            entry = trade.get("entry_conditions", {})
            if float(entry.get("volatility", 0.0)) > 0.012:
                tags.append("high volatility")
            else:
                tags.append("moderate volatility")

            momentum = float(entry.get("momentum", 0.0))
            if momentum > 0:
                tags.append("upward momentum")
            elif momentum < 0:
                tags.append("downward momentum")
            else:
                tags.append("flat momentum")

        # Pick top 2 most frequent conditions.
        counts: Dict[str, int] = {}
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
        return ", ".join([f"{name} ({count} mentions)" for name, count in top])

    # This function recommends tomorrow's focus using outcomes and learned insights.
    def _recommend_tomorrow_focus(
        self,
        trades_for_day: List[Dict[str, Any]],
        learned_insights: List[Dict[str, Any]],
    ) -> str:
        if not trades_for_day:
            return "Focus on patience and wait for high-confidence setups before entering."

        winners = [trade for trade in trades_for_day if trade.get("winner")]
        losers = [trade for trade in trades_for_day if trade.get("winner") is False]
        if len(losers) > len(winners):
            return (
                "Tomorrow, tighten entry quality: require stronger trend confirmation and avoid neutral-sentiment trades."
            )
        if learned_insights:
            latest = learned_insights[-1]
            text = str(latest.get("insight") or "").strip()
            if text:
                return f"Tomorrow, prioritize setups matching this learned pattern: {text}"
        return "Tomorrow, continue current risk discipline and favor setups with high AI confidence and aligned news."

    # This function stores one reasoning record and persists JSON + text logs.
    def _store_reason(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self.reasons.append(record)
            self._save_json()
            self._append_text_log(record)

    # This function loads existing JSON reasoning records from disk.
    def _load_reasons(self) -> None:
        try:
            if self.reasons_json_path.exists():
                with self.reasons_json_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        self.reasons = loaded
        except (OSError, json.JSONDecodeError):
            self.reasons = []

    # This function writes all structured reasoning records to trade_reasons.json.
    def _save_json(self) -> None:
        try:
            self.reasons_json_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reasons_json_path.open("w", encoding="utf-8") as handle:
                json.dump(self.reasons, handle, indent=2)
        except OSError:
            return

    # This function writes a human-readable log entry to trade_reasons.log.
    def _append_text_log(self, record: Dict[str, Any]) -> None:
        try:
            self.reasons_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.reasons_log_path.open("a", encoding="utf-8") as handle:
                handle.write(self._format_record_for_text(record))
                handle.write("\n\n" + ("-" * 90) + "\n\n")
        except OSError:
            return

    # This function formats one record as easy-to-read plain text for non-technical users.
    def _format_record_for_text(self, record: Dict[str, Any]) -> str:
        lines = [
            f"Trade ID: {record.get('trade_id')}",
            f"Record Type: {record.get('type')}",
            f"Timestamp: {record.get('timestamp')}",
            f"Instrument: {record.get('instrument', 'N/A')}",
            f"Direction: {record.get('direction', 'N/A')}",
            "",
            "Reasoning Details:",
        ]
        reasoning = record.get("reasoning", {})
        for section, content in reasoning.items():
            lines.append(f"  {section}:")
            lines.extend(self._format_text_value(content, indent_level=4))
        return "\n".join(lines)

    # This function recursively formats nested values so the text log stays human-readable.
    def _format_text_value(self, value: Any, indent_level: int) -> List[str]:
        indent = " " * indent_level
        lines: List[str] = []

        if isinstance(value, dict):
            for key, inner in value.items():
                lines.append(f"{indent}- {key}:")
                lines.extend(self._format_text_value(inner, indent_level + 2))
            return lines

        if isinstance(value, list):
            if not value:
                lines.append(f"{indent}- (none)")
                return lines
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    lines.extend(self._format_text_value(item, indent_level + 2))
                else:
                    lines.append(f"{indent}- {item}")
            return lines

        lines.append(f"{indent}- {value}")
        return lines
