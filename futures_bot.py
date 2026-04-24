"""Main QuantConnect futures bot wired to AI, risk, news, journaling, and reasoning."""

import json
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import pandas as pd
from AlgorithmImports import *

from ai_signal import AISignalEngine
from background_backtester import BackgroundBacktester
from config import (
    ACCOUNT_SIZE,
    DAILY_LOSS_LIMIT,
    TRAILING_DRAWDOWN_LIMIT,
    TRADERSPOST_WEBHOOK_URL,
)
from news_engine import NewsEngine
from risk_manager import RiskManager
from trade_journal import TradeJournal
from trade_reasoning import TradeReasoningEngine


class FuturesBot(QCAlgorithm):
    """Production wiring layer for all strategy intelligence modules."""

    # This function starts all modules and registers schedules/consolidators.
    def Initialize(self) -> None:
        self.SetStartDate(2024, 1, 1)
        self.SetCash(ACCOUNT_SIZE)
        self.SetTimeZone(TimeZones.NewYork)

        # Core trading settings that can be changed by background optimization.
        self.stop_loss_points = 10.0
        self.take_profit_points = 20.0
        self.live_parameters: Dict[str, float] = {
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "confidence_threshold": 0.60,
            "stop_loss_points": self.stop_loss_points,
            "take_profit_points": self.take_profit_points,
        }

        # Data and order tracking containers.
        self.training_days = 252
        self.max_five_minute_bars = 22000
        self.min_prediction_bars = 80
        self.future_states: Dict[Symbol, Dict[str, Any]] = {}
        self.pending_five_minute_bars: Deque[Tuple[Symbol, TradeBar]] = deque()
        self.order_webhook_payloads: Dict[int, Dict[str, Any]] = {}
        self.order_contexts: Dict[int, Dict[str, Any]] = {}
        self.last_forced_exit_reason = "manual"

        # Subscribe to ES and NQ for signal generation.
        es_symbol = self._add_continuous_future("ES", Futures.Indices.SP500EMini)
        nq_symbol = self._add_continuous_future("NQ", Futures.Indices.NASDAQ100EMini)

        # One account-level risk manager preserves all existing risk rules exactly.
        self.risk_manager = RiskManager(self, es_symbol)

        # Initialize journaling and reasoning engines.
        self.trade_journal = TradeJournal()
        self.trade_reasoning = TradeReasoningEngine()

        # Start news engine background fetch loop.
        self.news_engine = NewsEngine(algorithm=self, market_symbol=es_symbol)
        self.news_engine.start()

        # Start background backtester background loop.
        self.background_backtester = BackgroundBacktester(
            algorithm=self,
            es_symbol=es_symbol,
            nq_symbol=nq_symbol,
            get_live_parameters=self._get_live_parameters,
            apply_live_parameters=self._apply_live_parameters,
        )
        self.background_backtester.start()

        # Keep scheduled model retraining from previous implementation.
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.At(6, 0),
            self._scheduled_full_retrain,
        )

        # End-of-day post-processing run at 4:05 PM ET.
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(16, 5),
            self._run_end_of_day_maintenance,
        )

        backtester_status = self.background_backtester.get_status()
        news_status = self.news_engine.get_news_status()
        journal_status = self.trade_journal.get_rolling_stats()
        self.Debug(
            "All systems started | "
            f"background_backtester={backtester_status} | "
            f"news_engine={news_status} | "
            f"trade_journal={journal_status} | "
            "trade_reasoning=ready"
        )

    # This function processes each queued 5-minute bar using the full decision sequence.
    def OnData(self, data: Slice) -> None:
        while self.pending_five_minute_bars:
            canonical_symbol, bar = self.pending_five_minute_bars.popleft()
            self._process_trade_decision(canonical_symbol, bar)

    # This function handles order fills, closes journal records, and sends result webhooks.
    def OnOrderEvent(self, order_event: OrderEvent) -> None:
        if order_event.Status != OrderStatus.Filled:
            return

        order_payload = self.order_webhook_payloads.get(order_event.OrderId)
        if order_payload is None:
            action = "buy" if order_event.FillQuantity > 0 else "sell"
            order_payload = {
                "ticker": order_event.Symbol.Value,
                "action": action,
                "quantity": abs(int(order_event.FillQuantity)),
                "orderType": "market",
                "stopLoss": self.stop_loss_points,
                "takeProfit": self.take_profit_points,
            }
        self._send_webhook(order_payload)
        self.order_webhook_payloads.pop(order_event.OrderId, None)

        state = self._find_state_for_fill_symbol(order_event.Symbol)
        if state is None:
            return

        self._handle_bracket_fill_cleanup(state, order_event.OrderId)

        if not state.get("active_trade_id"):
            return

        if self._position_quantity_for_state(state) != 0:
            return

        exit_reason = self._infer_exit_reason(state, order_event)
        trade_id = str(state["active_trade_id"])
        closed_trade = self.trade_journal.record_trade_close(
            trade_id=trade_id,
            exit_time=self.Time,
            exit_price=float(order_event.FillPrice),
            exit_reason=exit_reason,
            point_value=self._point_value_for_ticker(state["ticker"]),
        )

        thesis_matched = bool(closed_trade.get("winner"))
        market_during_trade = {
            "price_move_points": float(closed_trade.get("pnl_points") or 0.0),
            "volatility_phase": "elevated" if abs(float(closed_trade.get("pnl_points") or 0.0)) > 6.0 else "calm",
            "momentum_shift": "favorable" if thesis_matched else "unfavorable",
            "pnl_points": float(closed_trade.get("pnl_points") or 0.0),
            "pnl_dollars": float(closed_trade.get("pnl_dollars") or 0.0),
        }
        self.trade_reasoning.generate_exit_reason(
            trade_id=trade_id,
            timestamp=self.Time,
            instrument=state["ticker"],
            direction=str(state.get("active_trade_direction", "unknown")),
            exit_type=exit_reason,
            market_during_trade=market_during_trade,
            thesis_matched=thesis_matched,
        )

        if self.trade_journal.confidence_low:
            self.Debug("Win rate dropped — reducing size until recovery")

        result_payload = {
            "ticker": state["ticker"],
            "action": "result",
            "quantity": int(closed_trade.get("contract_size") or 1),
            "orderType": "market",
            "stopLoss": self.stop_loss_points,
            "takeProfit": self.take_profit_points,
            "outcome": "win" if bool(closed_trade.get("winner")) else "loss",
            "pnl": round(float(closed_trade.get("pnl_dollars") or 0.0), 2),
            "exitReason": exit_reason,
        }
        self._send_webhook(result_payload)

        state["active_trade_id"] = None
        state["active_trade_direction"] = None
        state["active_trade_entry_price"] = None
        state["active_trade_entry_time"] = None
        state["active_trade_reason_summary"] = ""

    # This function shuts down background threads when the algorithm stops.
    def OnEndOfAlgorithm(self) -> None:
        if hasattr(self, "news_engine"):
            self.news_engine.stop()
        if hasattr(self, "background_backtester"):
            self.background_backtester.stop()

    # This function adds one continuous future stream and queues each 5-minute bar.
    def _add_continuous_future(self, ticker: str, future_type: Any) -> Symbol:
        future = self.AddFuture(future_type, Resolution.Minute)
        canonical_symbol = future.Symbol

        state: Dict[str, Any] = {
            "ticker": ticker,
            "canonical_symbol": canonical_symbol,
            "micro_canonical_symbol": self._try_add_micro_symbol(ticker),
            "engine": AISignalEngine(),
            "bars": deque(maxlen=self.max_five_minute_bars),
            "entry_ticket": None,
            "stop_ticket": None,
            "take_ticket": None,
            "active_trade_id": None,
            "active_trade_direction": None,
            "active_trade_entry_price": None,
            "active_trade_entry_time": None,
            "active_trade_reason_summary": "",
        }
        self.future_states[canonical_symbol] = state
        self._seed_history_from_lean(state)

        # Consolidator feeds bars into a queue; OnData executes the full decision flow.
        self.Consolidate(
            canonical_symbol,
            timedelta(minutes=5),
            lambda bar, symbol=canonical_symbol: self.pending_five_minute_bars.append((symbol, bar)),
        )
        return canonical_symbol

    # This function tries to subscribe to micro futures for half-size fallback.
    def _try_add_micro_symbol(self, ticker: str) -> Optional[Symbol]:
        candidates = {
            "ES": ["MicroSP500EMini", "MicroSP500Emini"],
            "NQ": ["MicroNASDAQ100EMini", "MicroNASDAQ100Emini"],
        }.get(ticker, [])

        for name in candidates:
            future_type = getattr(Futures.Indices, name, None)
            if future_type is None:
                continue
            try:
                return self.AddFuture(future_type, Resolution.Minute).Symbol
            except Exception:
                continue
        return None

    # This function runs the required 10-step decision flow for each 5-minute bar.
    def _process_trade_decision(self, canonical_symbol: Symbol, bar: TradeBar) -> None:
        state = self.future_states.get(canonical_symbol)
        if state is None:
            return
        ticker = state["ticker"]

        # Step 1: risk check must always happen first.
        if not self.risk_manager.check_risk():
            self.last_forced_exit_reason = "risk rule triggered"
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=0.0,
                news_score=self.news_engine.market_sentiment_score,
                reason="Risk manager blocked trading on this bar.",
            )
            return

        # Step 2: skip trading when news engine says current window is dangerous.
        now_utc = self._current_utc_time()
        avoid, avoid_reason = self.news_engine.should_avoid_trading(now_utc)
        if avoid:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=0.0,
                news_score=self.news_engine.market_sentiment_score,
                reason=f"News window block: {avoid_reason}",
            )
            return

        state["bars"].append({"time": bar.EndTime, "close": float(bar.Close), "volume": float(bar.Volume)})

        if state["engine"].retrain_needed(self.Time):
            self._train_symbol_model(canonical_symbol, state, reason="on-demand before trading")

        model_input = self._build_model_input(state, trading_days=60)
        if model_input is None:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=0.0,
                news_score=self.news_engine.market_sentiment_score,
                reason="Not enough recent history for AI prediction.",
            )
            return

        # Step 3: get signal from AI model.
        try:
            signal = state["engine"].predict(model_input)
            confidence = float(state["engine"].last_confidence)
        except ValueError as error:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=0.0,
                news_score=self.news_engine.market_sentiment_score,
                reason=f"AI prediction error: {error}",
            )
            return

        # Step 4: flat signal means do nothing.
        if signal == 0:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=confidence * 100.0,
                news_score=self.news_engine.market_sentiment_score,
                reason="AI returned a flat signal.",
            )
            return

        # Update watchdog timer only when a directional trade signal actually fires.
        self.risk_manager.record_signal_fired()

        # Step 5: fetch opportunity context and news alignment.
        opportunity = self.news_engine.find_opportunity(now_utc)
        news_sentiment = self._news_score_for_ticker(ticker)
        news_alignment = self._news_alignment_for_signal(signal, news_sentiment, opportunity)

        # Step 6: position sizing decision.
        upcoming_20 = self.news_engine._upcoming_no_trade_window(now_utc, lookahead_minutes=20)
        upcoming_30 = self.news_engine._upcoming_no_trade_window(now_utc, lookahead_minutes=30)
        has_20_30_event = upcoming_30 is not None and upcoming_20 is None

        if confidence < 0.55 or news_alignment == "contradict":
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=confidence * 100.0,
                news_score=news_sentiment,
                reason="Low confidence or contradicting news context.",
            )
            return

        if upcoming_20 is not None:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=confidence * 100.0,
                news_score=news_sentiment,
                reason=f"Upcoming no-trade event in <20m: {upcoming_20.name}",
            )
            return

        size_mode = "full"
        if (
            0.55 <= confidence <= 0.65
            or news_alignment == "neutral"
            or has_20_30_event
            or self.trade_journal.confidence_low
        ):
            size_mode = "half"

        if (
            confidence > 0.65
            and news_alignment == "agree"
            and upcoming_20 is None
            and not self.trade_journal.confidence_low
        ):
            size_mode = "full"

        mapped_symbol, quantity, sizing_note = self._resolve_execution_contract(state, size_mode=size_mode)
        if mapped_symbol is None or quantity <= 0:
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=confidence * 100.0,
                news_score=news_sentiment,
                reason=f"Cannot execute selected size mode: {sizing_note}",
            )
            return

        # Avoid stacking duplicate positions on same ticker.
        net_qty = self._position_quantity_for_state(state)
        desired_direction = 1 if signal > 0 else -1
        if net_qty != 0 and (1 if net_qty > 0 else -1) == desired_direction:
            self._print_decision_line(
                ticker=ticker,
                action="HOLD",
                confidence_pct=confidence * 100.0,
                news_score=news_sentiment,
                reason="Position already open in the same direction.",
            )
            return
        if net_qty != 0 and (1 if net_qty > 0 else -1) != desired_direction:
            self.last_forced_exit_reason = "signal reversed"
            self.Liquidate(self._current_mapped_symbol(state["canonical_symbol"]), "signal reversed")
            if state.get("micro_canonical_symbol"):
                micro_mapped = self._current_mapped_symbol(state["micro_canonical_symbol"])
                if micro_mapped is not None:
                    self.Liquidate(micro_mapped, "signal reversed")
            self._print_decision_line(
                ticker=ticker,
                action="SKIP",
                confidence_pct=confidence * 100.0,
                news_score=news_sentiment,
                reason="Signal reversed while position was open; flattening first.",
            )
            return

        # Step 7: generate full reasoning block for this entry.
        trade_id = str(uuid.uuid4())
        technicals = self._build_technicals_for_reasoning(model_input)
        ai_context = {
            "confidence": confidence,
            "top_features": self._top_model_features(state["engine"]),
            "historical_similarity_text": self._historical_similarity_text(confidence),
            "in_pattern": bool(confidence >= 0.65 and news_alignment != "contradict"),
        }
        news_context = {
            "market_sentiment_score": news_sentiment,
            "relevant_headlines": self._recent_relevant_headlines(ticker),
            "supports_signal": news_alignment == "agree",
            "upcoming_events": self._list_upcoming_events(now_utc, 30),
        }
        point_value = self._point_value_for_ticker(ticker)
        direction_text = "long" if signal > 0 else "short"
        risk_context = {
            "entry_price": float(bar.Close),
            "stop_loss": float(bar.Close - self.stop_loss_points if signal > 0 else bar.Close + self.stop_loss_points),
            "take_profit": float(bar.Close + self.take_profit_points if signal > 0 else bar.Close - self.take_profit_points),
            "point_value": point_value,
            "quantity": quantity,
            "daily_pnl": float(self.risk_manager.daily_pnl),
            "daily_loss_limit": float(DAILY_LOSS_LIMIT),
            "trailing_drawdown": float(self.risk_manager.trailing_drawdown),
            "trailing_drawdown_limit": float(TRAILING_DRAWDOWN_LIMIT),
        }
        entry_reason = self.trade_reasoning.generate_entry_reason(
            trade_id=trade_id,
            timestamp=self.Time,
            instrument=ticker,
            direction=direction_text,
            technicals=technicals,
            ai_context=ai_context,
            news_context=news_context,
            risk_context=risk_context,
        )
        reason_sentence = str(entry_reason["reasoning"]["CONFIDENCE VERDICT"]["summary"])

        # Step 8: one-line console summary for every decision.
        action_label = "BUY_FULL" if signal > 0 and size_mode == "full" else (
            "BUY_HALF" if signal > 0 else ("SELL_FULL" if size_mode == "full" else "SELL_HALF")
        )
        self._print_decision_line(
            ticker=ticker,
            action=action_label,
            confidence_pct=confidence * 100.0,
            news_score=news_sentiment,
            reason=reason_sentence,
        )

        # Step 9: place entry and bracket orders.
        self._submit_bracket_entry(
            state=state,
            mapped_symbol=mapped_symbol,
            direction=signal,
            quantity=quantity,
            reference_price=float(bar.Close),
            trade_id=trade_id,
            reason_summary=reason_sentence,
            size_mode=size_mode,
            confidence=confidence,
            news_sentiment=news_sentiment,
        )

        # Step 10: record trade open in journal.
        active_parameter_set = self._get_live_parameters()
        self.trade_journal.record_trade_open(
            entry_time=self.Time,
            instrument=ticker,
            direction=direction_text,
            entry_price=float(bar.Close),
            contract_size=quantity,
            rsi_value=float(technicals["rsi"]),
            ma_fast_value=float(technicals["ma_fast"]),
            ma_slow_value=float(technicals["ma_slow"]),
            momentum_value=float(technicals["momentum"]),
            volatility_value=float(technicals["volatility"]),
            volume_value=float(technicals["volume_relative"]),
            news_sentiment_score=float(news_sentiment),
            ai_confidence_score=float(confidence),
            reasoning_text=reason_sentence,
            stop_loss_level=float(risk_context["stop_loss"]),
            take_profit_level=float(risk_context["take_profit"]),
            active_parameter_set=active_parameter_set,
            trade_id=trade_id,
        )

    # This function trains all models weekly and includes journal feedback visibility.
    def _scheduled_full_retrain(self) -> None:
        feedback = self.trade_journal.consume_pending_training_feedback()
        if feedback:
            self.Debug(f"[AI] Applying journal feedback to next retrain batch: {len(feedback)} samples")

        for canonical_symbol, state in self.future_states.items():
            self._train_symbol_model(canonical_symbol, state, reason="scheduled Monday 06:00 retrain")

    # This function performs end-of-day summary/logging/upgrade checks.
    def _run_end_of_day_maintenance(self) -> None:
        now_utc = self._current_utc_time()
        day_trades = self._closed_trades_for_day(now_utc.date())
        insights = list(self.trade_journal.learned_insights)
        summary = self.trade_reasoning.generate_daily_summary(
            day=self.Time.date(),
            trades_for_day=day_trades,
            learned_insights=insights,
        )

        backtest_status = self.background_backtester.get_status()
        if bool(backtest_status.get("upgrade_pending")):
            # Use the backtester's own promotion safety checks (flat time, consistency, risk limits).
            self.background_backtester._attempt_promotion_if_safe()
            backtest_status = self.background_backtester.get_status()

        latest_insights = [row.get("insight", "") for row in insights[-3:]]
        self.Debug(
            f"[EOD] Daily summary stored. Backtester status={backtest_status}. "
            f"Learned insights={latest_insights}. "
            f"Trades today={len(day_trades)}."
        )

        self._print_decision_line(
            ticker="SYSTEM",
            action="EOD",
            confidence_pct=0.0,
            news_score=self.news_engine.market_sentiment_score,
            reason="End-of-day summary completed and module health checked.",
        )

        # Keep linter happy that summary is intentionally produced and stored.
        _ = summary

    # This function trains one symbol model using recent 5-minute bars.
    def _train_symbol_model(self, canonical_symbol: Symbol, state: Dict[str, Any], reason: str) -> None:
        model_input = self._build_model_input(state, trading_days=self.training_days)
        if model_input is None:
            self.Debug(f"[AI] Skipping retrain for {state['ticker']} - not enough history.")
            return
        try:
            state["engine"].train(model_input)
            self.Debug(f"[AI] Retrained {state['ticker']} model ({len(model_input)} rows) because {reason}.")
        except ValueError as error:
            self.Debug(f"[AI] Retrain failed for {state['ticker']}: {error}")

    # This function seeds initial 5-minute history from Lean minute bars.
    def _seed_history_from_lean(self, state: Dict[str, Any]) -> None:
        history = self.History(state["canonical_symbol"], timedelta(days=320), Resolution.Minute)
        history_frame = self._normalize_history_frame(history, state["canonical_symbol"])
        if history_frame is None or history_frame.empty:
            return

        five_minute = (
            history_frame[["close", "volume"]]
            .resample("5min")
            .agg({"close": "last", "volume": "sum"})
            .dropna()
        )
        for time_index, row in five_minute.iterrows():
            state["bars"].append(
                {"time": time_index, "close": float(row["close"]), "volume": float(row["volume"])}
            )

    # This function builds model input DataFrame from cached 5-minute bars.
    def _build_model_input(self, state: Dict[str, Any], trading_days: int) -> Optional[pd.DataFrame]:
        bars = state["bars"]
        if not bars:
            return None
        frame = pd.DataFrame(list(bars))
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.drop_duplicates(subset=["time"], keep="last").sort_values("time").set_index("time")
        if frame.empty:
            return None
        cutoff = frame.index.max() - pd.tseries.offsets.BDay(trading_days)
        filtered = frame.loc[frame.index >= cutoff, ["close", "volume"]].dropna()
        if len(filtered) < self.min_prediction_bars:
            return None
        return filtered

    # This function resolves whether to use primary or micro contract for size control.
    def _resolve_execution_contract(
        self, state: Dict[str, Any], size_mode: str
    ) -> Tuple[Optional[Symbol], int, str]:
        primary_mapped = self._current_mapped_symbol(state["canonical_symbol"])
        if primary_mapped is None:
            return None, 0, "Primary mapped symbol unavailable."

        if size_mode == "full":
            return primary_mapped, 1, "Using primary contract."

        micro_canonical = state.get("micro_canonical_symbol")
        if micro_canonical is not None:
            micro_mapped = self._current_mapped_symbol(micro_canonical)
            if micro_mapped is not None:
                return micro_mapped, 1, "Using micro contract for reduced size."
        return None, 0, "Half-size mode requested but no micro contract is available."

    # This function submits entry + stop + target orders and links them to trade metadata.
    def _submit_bracket_entry(
        self,
        state: Dict[str, Any],
        mapped_symbol: Symbol,
        direction: int,
        quantity: int,
        reference_price: float,
        trade_id: str,
        reason_summary: str,
        size_mode: str,
        confidence: float,
        news_sentiment: float,
    ) -> None:
        self._cancel_exit_orders(state)

        signed_qty = quantity if direction > 0 else -quantity
        entry_tag = f"{state['ticker']} AI entry ({size_mode})"
        entry_ticket = self.MarketOrder(mapped_symbol, signed_qty, tag=entry_tag)

        if direction > 0:
            stop_price = reference_price - self.stop_loss_points
            target_price = reference_price + self.take_profit_points
            exit_qty = -quantity
            action = "buy"
            direction_text = "long"
        else:
            stop_price = reference_price + self.stop_loss_points
            target_price = reference_price - self.take_profit_points
            exit_qty = quantity
            action = "sell"
            direction_text = "short"

        stop_ticket = self.StopMarketOrder(
            mapped_symbol,
            exit_qty,
            stop_price,
            tag=f"{state['ticker']} stop loss",
        )
        take_ticket = self.LimitOrder(
            mapped_symbol,
            exit_qty,
            target_price,
            tag=f"{state['ticker']} take profit",
        )

        state["entry_ticket"] = entry_ticket
        state["stop_ticket"] = stop_ticket
        state["take_ticket"] = take_ticket
        state["active_trade_id"] = trade_id
        state["active_trade_direction"] = direction_text
        state["active_trade_entry_price"] = reference_price
        state["active_trade_entry_time"] = self.Time
        state["active_trade_reason_summary"] = reason_summary

        self.order_contexts[entry_ticket.OrderId] = {"trade_id": trade_id, "role": "entry", "state": state}
        self.order_contexts[stop_ticket.OrderId] = {"trade_id": trade_id, "role": "stop", "state": state}
        self.order_contexts[take_ticket.OrderId] = {"trade_id": trade_id, "role": "target", "state": state}

        self._register_order_webhook_payload(
            entry_ticket.OrderId,
            state["ticker"],
            action,
            quantity,
            self.stop_loss_points,
            self.take_profit_points,
        )
        self._register_order_webhook_payload(
            stop_ticket.OrderId,
            state["ticker"],
            "sell" if direction > 0 else "buy",
            quantity,
            self.stop_loss_points,
            self.take_profit_points,
        )
        self._register_order_webhook_payload(
            take_ticket.OrderId,
            state["ticker"],
            "sell" if direction > 0 else "buy",
            quantity,
            self.stop_loss_points,
            self.take_profit_points,
        )

        # Keep these values for result notifications.
        state["active_trade_confidence"] = confidence
        state["active_trade_news_sentiment"] = news_sentiment

    # This function cancels stop/target tickets to avoid stale exits.
    def _cancel_exit_orders(self, state: Dict[str, Any]) -> None:
        for key in ("stop_ticket", "take_ticket"):
            ticket = state.get(key)
            if ticket is None:
                continue
            if ticket.Status in (OrderStatus.New, OrderStatus.Submitted, OrderStatus.PartiallyFilled):
                ticket.Cancel("Replacing bracket orders")
            state[key] = None

    # This function cancels sibling exit orders after stop/target fills.
    def _handle_bracket_fill_cleanup(self, state: Dict[str, Any], filled_order_id: int) -> None:
        stop_ticket = state.get("stop_ticket")
        take_ticket = state.get("take_ticket")
        stop_id = stop_ticket.OrderId if stop_ticket is not None else None
        take_id = take_ticket.OrderId if take_ticket is not None else None

        if filled_order_id == stop_id:
            if take_ticket is not None and take_ticket.Status in (
                OrderStatus.New,
                OrderStatus.Submitted,
                OrderStatus.PartiallyFilled,
            ):
                take_ticket.Cancel("Stop filled, cancel target")
            state["stop_ticket"] = None
            state["take_ticket"] = None
            return

        if filled_order_id == take_id:
            if stop_ticket is not None and stop_ticket.Status in (
                OrderStatus.New,
                OrderStatus.Submitted,
                OrderStatus.PartiallyFilled,
            ):
                stop_ticket.Cancel("Target filled, cancel stop")
            state["stop_ticket"] = None
            state["take_ticket"] = None

    # This function maps filled symbols back to the corresponding state.
    def _find_state_for_fill_symbol(self, fill_symbol: Symbol) -> Optional[Dict[str, Any]]:
        for state in self.future_states.values():
            primary = self._current_mapped_symbol(state["canonical_symbol"])
            micro = None
            if state.get("micro_canonical_symbol") is not None:
                micro = self._current_mapped_symbol(state["micro_canonical_symbol"])
            if fill_symbol == primary or (micro is not None and fill_symbol == micro):
                return state
        return None

    # This function infers why a trade closed using order role and recent context.
    def _infer_exit_reason(self, state: Dict[str, Any], order_event: OrderEvent) -> str:
        context = self.order_contexts.get(order_event.OrderId, {})
        role = context.get("role")
        if role == "stop":
            return "stop hit"
        if role == "target":
            return "target hit"

        order = self.Transactions.GetOrderById(order_event.OrderId)
        tag = str(order.Tag if order is not None else "").lower()
        if "eod" in tag:
            return "EOD flatten"
        if "signal reversed" in tag:
            return "signal reversed"
        if "news" in tag:
            return "news event"
        if self.last_forced_exit_reason == "risk rule triggered":
            return "risk rule triggered"
        return "manual"

    # This function builds one-line decision output for trade/skip transparency.
    def _print_decision_line(
        self,
        ticker: str,
        action: str,
        confidence_pct: float,
        news_score: float,
        reason: str,
    ) -> None:
        self.Debug(
            f"[{self.Time.strftime('%Y-%m-%d %H:%M:%S')}] [{ticker}] [{action}] | "
            f"Confidence: {confidence_pct:.1f}% | "
            f"News: {news_score:+.2f} | "
            f"Reason: {reason} | "
            f"P&L today: ${self.risk_manager.daily_pnl:.2f}"
        )

    # This function returns live parameter values for background optimization callbacks.
    def _get_live_parameters(self) -> Dict[str, float]:
        return dict(self.live_parameters)

    # This function applies promoted parameters safely to live bot settings.
    def _apply_live_parameters(self, parameters: Dict[str, float]) -> None:
        self.live_parameters = dict(parameters)
        self.stop_loss_points = float(parameters.get("stop_loss_points", self.stop_loss_points))
        self.take_profit_points = float(parameters.get("take_profit_points", self.take_profit_points))
        self.Debug(f"[PARAMS] Applied promoted live parameters: {self.live_parameters}")

    # This function returns primary mapped symbol if available.
    def _current_mapped_symbol(self, canonical_symbol: Symbol) -> Optional[Symbol]:
        if canonical_symbol not in self.Securities:
            return None
        mapped = self.Securities[canonical_symbol].Mapped
        if mapped is None:
            return None
        return mapped

    # This function returns current net quantity across primary + micro contracts.
    def _position_quantity_for_state(self, state: Dict[str, Any]) -> int:
        quantity = 0
        primary = self._current_mapped_symbol(state["canonical_symbol"])
        if primary is not None:
            quantity += int(self.Portfolio[primary].Quantity)
        micro_canonical = state.get("micro_canonical_symbol")
        if micro_canonical is not None:
            micro = self._current_mapped_symbol(micro_canonical)
            if micro is not None:
                quantity += int(self.Portfolio[micro].Quantity)
        return quantity

    # This function scores news alignment versus signal direction.
    def _news_alignment_for_signal(
        self, signal: int, news_sentiment: float, opportunity: Optional[Dict[str, Any]]
    ) -> str:
        if opportunity is not None:
            direction = str(opportunity.get("direction", "")).lower()
            if (signal > 0 and direction == "long") or (signal < 0 and direction == "short"):
                return "agree"
        if signal > 0 and news_sentiment > 0.10:
            return "agree"
        if signal < 0 and news_sentiment < -0.10:
            return "agree"
        if abs(news_sentiment) <= 0.10:
            return "neutral"
        return "contradict"

    # This function returns the most relevant sentiment stream for each ticker.
    def _news_score_for_ticker(self, ticker: str) -> float:
        if ticker == "ES":
            return float(self.news_engine.es_sentiment_score)
        return float(self.news_engine.market_sentiment_score)

    # This function extracts simplified market conditions for journaling and reasoning.
    def _build_technicals_for_reasoning(self, model_input: pd.DataFrame) -> Dict[str, float]:
        close = model_input["close"]
        volume = model_input["volume"]
        rsi = self._calculate_rsi(close, period=14).iloc[-1]
        ma_fast = close.rolling(20).mean().iloc[-1]
        ma_slow = close.rolling(50).mean().iloc[-1]
        momentum = close.pct_change(5).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        volume_relative = (volume.iloc[-1] / max(1e-9, float(volume.rolling(20).mean().iloc[-1])))

        return {
            "rsi": float(0.0 if pd.isna(rsi) else rsi),
            "ma_fast": float(0.0 if pd.isna(ma_fast) else ma_fast),
            "ma_slow": float(0.0 if pd.isna(ma_slow) else ma_slow),
            "momentum": float(0.0 if pd.isna(momentum) else momentum),
            "volatility": float(0.0 if pd.isna(volatility) else volatility),
            "volume_relative": float(1.0 if pd.isna(volume_relative) else volume_relative),
        }

    # This function calculates RSI for reasoning/journal context values.
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, pd.NA)
        return 100.0 - (100.0 / (1.0 + rs))

    # This function returns model feature names ordered by importance.
    def _top_model_features(self, engine: AISignalEngine) -> List[str]:
        names = ["rsi_14", "ma_20_50_diff", "momentum_5", "volatility_20", "volume_relative_20"]
        model = getattr(engine, "model", None)
        if model is None or not hasattr(model, "feature_importances_"):
            return names[:3]
        importances = list(getattr(model, "feature_importances_"))
        ranked = sorted(zip(names, importances), key=lambda row: row[1], reverse=True)
        return [f"{name} ({weight:.3f})" for name, weight in ranked[:3]]

    # This function creates a short similarity message from confidence level.
    def _historical_similarity_text(self, confidence: float) -> str:
        if confidence >= 0.75:
            return "This setup is close to high-confidence historical winners."
        if confidence >= 0.60:
            return "This setup is moderately similar to prior mixed-win trades."
        return "This setup has weak historical similarity and should be treated cautiously."

    # This function returns recent headlines relevant for a specific instrument.
    def _recent_relevant_headlines(self, ticker: str) -> List[str]:
        rows = list(self.news_engine.news_log[-50:])
        headlines: List[str] = []
        for row in reversed(rows):
            text = str(row.get("headline", ""))
            text_lower = text.lower()
            if ticker == "ES":
                if any(key in text_lower for key in ("s&p", "spy", "fed", "economy", "cpi", "gdp")):
                    headlines.append(text)
            else:
                if any(key in text_lower for key in ("nasdaq", "qqq", "tech", "fed", "economy")):
                    headlines.append(text)
            if len(headlines) >= 3:
                break
        return headlines

    # This function lists upcoming no-trade events for reasoning context.
    def _list_upcoming_events(self, now_utc: datetime, lookahead_minutes: int) -> List[str]:
        window = self.news_engine._upcoming_no_trade_window(now_utc, lookahead_minutes=lookahead_minutes)
        if window is None:
            return []
        return [window.name]

    # This function fetches closed trades for the requested UTC day.
    def _closed_trades_for_day(self, day_utc_date: datetime.date) -> List[Dict[str, Any]]:
        trades: List[Dict[str, Any]] = []
        for trade in self.trade_journal.trades:
            if trade.get("status") != "closed":
                continue
            exit_time = trade.get("exit_time")
            if not exit_time:
                continue
            try:
                dt = datetime.fromisoformat(str(exit_time).replace("Z", "+00:00")).astimezone(timezone.utc)
            except ValueError:
                continue
            if dt.date() == day_utc_date:
                trades.append(trade)
        return trades

    # This function returns point value for futures contracts for P&L/risk calculations.
    def _point_value_for_ticker(self, ticker: str) -> float:
        return {"ES": 50.0, "NQ": 20.0}.get(str(ticker).upper(), 1.0)

    # This function returns a reliable UTC timestamp across Lean runtime datetime types.
    def _current_utc_time(self) -> datetime:
        candidate = getattr(self, "UtcTime", None) or self.Time
        if candidate.tzinfo is None:
            return candidate.replace(tzinfo=timezone.utc)
        return candidate.astimezone(timezone.utc)

    # This function registers per-order payloads for TradersPost.
    def _register_order_webhook_payload(
        self,
        order_id: int,
        ticker: str,
        action: str,
        quantity: int,
        stop_loss_points: float,
        take_profit_points: float,
    ) -> None:
        self.order_webhook_payloads[order_id] = {
            "ticker": ticker,
            "action": action,
            "quantity": int(quantity),
            "orderType": "market",
            "stopLoss": float(stop_loss_points),
            "takeProfit": float(take_profit_points),
        }

    # This function posts webhook payloads safely without stopping trading on network errors.
    def _send_webhook(self, payload: Dict[str, Any]) -> None:
        if not TRADERSPOST_WEBHOOK_URL:
            return
        try:
            self.Notify.Web(TRADERSPOST_WEBHOOK_URL, json.dumps(payload))
        except Exception as error:
            self.Debug(f"[WEBHOOK] Failed to send payload: {error} | payload={payload}")

    # This function sends manual TradersPost test payloads without placing trades.
    def SendTestTradersPostSignal(self, ticker: str = "ES", action: str = "buy", quantity: int = 1) -> None:
        payload = {
            "ticker": ticker,
            "action": action,
            "quantity": int(quantity),
            "orderType": "market",
            "stopLoss": float(self.stop_loss_points),
            "takeProfit": float(self.take_profit_points),
        }
        self.Debug(f"[WEBHOOK] Sending manual test payload: {payload}")
        self._send_webhook(payload)

    # This function normalizes Lean history output into close/volume DataFrame columns.
    def _normalize_history_frame(self, history: pd.DataFrame, symbol: Symbol) -> Optional[pd.DataFrame]:
        if history is None or history.empty:
            return None
        frame = history
        if "symbol" in getattr(frame.index, "names", []):
            frame = frame.xs(symbol, level="symbol")
        renamed = frame.rename(columns={"Close": "close", "Volume": "volume"})
        lower_cols = {str(col).lower(): col for col in renamed.columns}
        if "close" not in lower_cols or "volume" not in lower_cols:
            return None
        selected = renamed[[lower_cols["close"], lower_cols["volume"]]].copy()
        selected.columns = ["close", "volume"]
        selected["close"] = pd.to_numeric(selected["close"], errors="coerce")
        selected["volume"] = pd.to_numeric(selected["volume"], errors="coerce")
        return selected.dropna()
