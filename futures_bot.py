"""Main QuantConnect futures algorithm that wires AI + risk controls together."""

import json
from collections import deque
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
from AlgorithmImports import *

from ai_signal import AISignalEngine
from config import ACCOUNT_SIZE, TRADERSPOST_WEBHOOK_URL
from risk_manager import RiskManager


class FuturesBot(QCAlgorithm):
    """Complete algorithm that trades ES and NQ with AI signals and risk checks."""

    # This function sets up subscriptions, model engines, schedules, and risk controls.
    def Initialize(self) -> None:
        self.SetStartDate(2024, 1, 1)
        self.SetCash(ACCOUNT_SIZE)
        self.SetTimeZone(TimeZones.NewYork)

        # These are the fixed trade management distances requested by the strategy spec.
        self.stop_loss_points = 10.0
        self.take_profit_points = 20.0

        # We keep enough 5-minute bars for feature generation and long-horizon retraining.
        self.training_days = 252
        self.max_five_minute_bars = 22000
        self.min_prediction_bars = 80
        self.order_webhook_payloads: Dict[int, Dict] = {}

        # Add both requested continuous futures contracts.
        self.future_states: Dict[Symbol, Dict] = {}
        es_symbol = self._add_continuous_future("ES", Futures.Indices.SP500EMini)
        self._add_continuous_future("NQ", Futures.Indices.NASDAQ100EMini)

        # One account-level risk manager is enough because P&L and drawdown are account-wide.
        self.risk_manager = RiskManager(self, es_symbol)

        # Retrain all models weekly every Monday at 06:00 as requested.
        self.Schedule.On(
            self.DateRules.Every(DayOfWeek.Monday),
            self.TimeRules.At(6, 0),
            self._scheduled_full_retrain,
        )

    # This function is intentionally empty because trade logic runs on 5-minute consolidators.
    def OnData(self, data: Slice) -> None:
        return

    # This function sends webhook notifications after orders fill.
    def OnOrderEvent(self, order_event: OrderEvent) -> None:
        if order_event.Status not in (OrderStatus.Filled, OrderStatus.PartiallyFilled):
            return

        payload = self.order_webhook_payloads.get(order_event.OrderId)
        if payload is None:
            action = "buy" if order_event.FillQuantity > 0 else "sell"
            payload = {
                "ticker": order_event.Symbol.Value,
                "action": action,
                "quantity": abs(int(order_event.FillQuantity)),
                "stopLoss": None,
                "takeProfit": None,
            }

        self._send_webhook(payload)

        if order_event.Status == OrderStatus.Filled and order_event.OrderId in self.order_webhook_payloads:
            self.order_webhook_payloads.pop(order_event.OrderId, None)

    # This function adds a continuous future contract and wires a 5-minute bar handler.
    def _add_continuous_future(self, ticker: str, future_type) -> Symbol:
        future = self.AddFuture(future_type, Resolution.Minute)
        future_symbol = future.Symbol

        state = {
            "ticker": ticker,
            "canonical_symbol": future_symbol,
            "engine": AISignalEngine(),
            "bars": deque(maxlen=self.max_five_minute_bars),
            "stop_ticket": None,
            "take_ticket": None,
        }
        self.future_states[future_symbol] = state
        self._seed_history_from_lean(state)

        # This creates true 5-minute trading bars from minute data.
        self.Consolidate(
            future_symbol,
            timedelta(minutes=5),
            lambda bar, symbol=future_symbol: self._on_five_minute_bar(symbol, bar),
        )
        return future_symbol

    # This function fills each symbol state with initial 5-minute history from Lean.
    def _seed_history_from_lean(self, state: Dict) -> None:
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

    # This function runs the complete decision loop each time a new 5-minute bar arrives.
    def _on_five_minute_bar(self, canonical_symbol: Symbol, bar: TradeBar) -> None:
        # Rule: risk checks must run first, and if they fail, we do nothing else.
        if not self.risk_manager.check_risk():
            return

        state = self.future_states[canonical_symbol]
        state["bars"].append({"time": bar.EndTime, "close": float(bar.Close), "volume": float(bar.Volume)})

        # Retrain on demand before any prediction or order placement.
        if state["engine"].retrain_needed(self.Time):
            self._train_symbol_model(canonical_symbol, state, reason="on-demand before trade")

        model_input = self._build_model_input(state, trading_days=60)
        if model_input is None:
            return

        try:
            signal = state["engine"].predict(model_input)
            confidence = float(state["engine"].last_confidence)
        except ValueError as error:
            self.Debug(f"[AI] Prediction failed for {state['ticker']}: {error}")
            return

        if signal in (1, -1):
            self.risk_manager.record_signal_fired()

        self._log_trade_decision(state["ticker"], signal, confidence)
        self._apply_signal_to_position(state, signal, bar)

    # This function retrains both symbol models every Monday at 06:00.
    def _scheduled_full_retrain(self) -> None:
        for canonical_symbol, state in self.future_states.items():
            self._train_symbol_model(canonical_symbol, state, reason="scheduled Monday 06:00 retrain")

    # This function trains one symbol model using the last 252 trading days of 5-minute data.
    def _train_symbol_model(self, canonical_symbol: Symbol, state: Dict, reason: str) -> None:
        model_input = self._build_model_input(state, trading_days=self.training_days)
        if model_input is None:
            self.Debug(f"[AI] Skipping retrain for {state['ticker']} - not enough history.")
            return

        try:
            state["engine"].train(model_input)
            self.Debug(
                f"[AI] Retrained {state['ticker']} model ({len(model_input)} rows) because {reason}."
            )
        except ValueError as error:
            self.Debug(f"[AI] Retrain failed for {state['ticker']}: {error}")

    # This function builds a clean DataFrame from cached 5-minute bars for model use.
    def _build_model_input(self, state: Dict, trading_days: int) -> Optional[pd.DataFrame]:
        if not state["bars"]:
            return None

        frame = pd.DataFrame(list(state["bars"]))
        frame["time"] = pd.to_datetime(frame["time"])
        frame = frame.drop_duplicates(subset=["time"], keep="last").sort_values("time")
        frame = frame.set_index("time")

        if frame.empty:
            return None

        cutoff = frame.index.max() - pd.tseries.offsets.BDay(trading_days)
        filtered = frame.loc[frame.index >= cutoff, ["close", "volume"]].dropna()
        if len(filtered) < self.min_prediction_bars:
            return None
        return filtered

    # This function places the requested long/short/flat orders plus stop and target brackets.
    def _apply_signal_to_position(self, state: Dict, signal: int, bar: TradeBar) -> None:
        mapped_symbol = self._current_mapped_symbol(state["canonical_symbol"])
        if mapped_symbol is None:
            return

        current_quantity = int(self.Portfolio[mapped_symbol].Quantity)
        if signal == 0:
            self._cancel_exit_orders(state)
            if current_quantity != 0:
                self.Liquidate(mapped_symbol, "AI signal is flat")
            return

        if signal == 1:
            if current_quantity < 0:
                self._cancel_exit_orders(state)
                self.Liquidate(mapped_symbol, "Flip from short to long")
                return
            if current_quantity == 0:
                self._submit_bracket_entry(state, mapped_symbol, 1, float(bar.Close))
            return

        if signal == -1:
            if current_quantity > 0:
                self._cancel_exit_orders(state)
                self.Liquidate(mapped_symbol, "Flip from long to short")
                return
            if current_quantity == 0:
                self._submit_bracket_entry(state, mapped_symbol, -1, float(bar.Close))

    # This function submits entry + stop-loss + take-profit orders for one contract.
    def _submit_bracket_entry(self, state: Dict, mapped_symbol: Symbol, direction: int, reference_price: float) -> None:
        self._cancel_exit_orders(state)

        entry_ticket = self.MarketOrder(mapped_symbol, direction, tag=f"{state['ticker']} AI entry")
        if direction > 0:
            stop_price = reference_price - self.stop_loss_points
            take_profit_price = reference_price + self.take_profit_points
            exit_quantity = -1
            action = "buy"
        else:
            stop_price = reference_price + self.stop_loss_points
            take_profit_price = reference_price - self.take_profit_points
            exit_quantity = 1
            action = "sell"

        stop_ticket = self.StopMarketOrder(
            mapped_symbol, exit_quantity, stop_price, tag=f"{state['ticker']} stop loss"
        )
        take_ticket = self.LimitOrder(
            mapped_symbol, exit_quantity, take_profit_price, tag=f"{state['ticker']} take profit"
        )
        state["stop_ticket"] = stop_ticket
        state["take_ticket"] = take_ticket

        self._register_order_webhook_payload(
            entry_ticket.OrderId,
            state["ticker"],
            action,
            1,
            stop_price,
            take_profit_price,
        )
        self._register_order_webhook_payload(
            stop_ticket.OrderId,
            state["ticker"],
            "sell" if direction > 0 else "buy",
            1,
            stop_price,
            take_profit_price,
        )
        self._register_order_webhook_payload(
            take_ticket.OrderId,
            state["ticker"],
            "sell" if direction > 0 else "buy",
            1,
            stop_price,
            take_profit_price,
        )

    # This function cancels any old stop/target tickets so stale exits do not fire.
    def _cancel_exit_orders(self, state: Dict) -> None:
        for key in ("stop_ticket", "take_ticket"):
            ticket = state.get(key)
            if ticket is None:
                continue
            if ticket.Status in (OrderStatus.New, OrderStatus.Submitted, OrderStatus.PartiallyFilled):
                ticket.Cancel("Replacing bracket orders")
            state[key] = None

    # This function records a webhook payload for later send when the order fills.
    def _register_order_webhook_payload(
        self,
        order_id: int,
        ticker: str,
        action: str,
        quantity: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> None:
        self.order_webhook_payloads[order_id] = {
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "stopLoss": stop_loss,
            "takeProfit": take_profit,
        }

    # This function sends the trade payload to TraderPost as JSON when a webhook URL exists.
    def _send_webhook(self, payload: Dict) -> None:
        if not TRADERSPOST_WEBHOOK_URL:
            return
        try:
            self.Notify.Web(TRADERSPOST_WEBHOOK_URL, json.dumps(payload))
        except Exception as error:
            self.Debug(f"[WEBHOOK] Failed to send payload: {error}")

    # This function logs all required decision fields for auditability and debugging.
    def _log_trade_decision(self, ticker: str, signal: int, confidence: float) -> None:
        self.Debug(
            "[DECISION] "
            f"time={self.Time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"ticker={ticker} "
            f"signal={signal} "
            f"confidence={confidence:.3f} "
            f"dailyPnL={self.risk_manager.daily_pnl:.2f} "
            f"drawdown={self.risk_manager.trailing_drawdown:.2f}"
        )

    # This function returns the currently tradable mapped contract for a continuous future.
    def _current_mapped_symbol(self, canonical_symbol: Symbol) -> Optional[Symbol]:
        if not self.Securities.ContainsKey(canonical_symbol):
            return None
        mapped_symbol = self.Securities[canonical_symbol].Mapped
        if mapped_symbol is None:
            return None
        return mapped_symbol

    # This function normalizes Lean history output into lowercase close/volume DataFrame columns.
    def _normalize_history_frame(self, history: pd.DataFrame, symbol: Symbol) -> Optional[pd.DataFrame]:
        if history is None or history.empty:
            return None

        frame = history
        if "symbol" in getattr(frame.index, "names", []):
            frame = frame.xs(symbol, level="symbol")

        renamed = frame.rename(columns={"Close": "close", "Volume": "volume"})
        normalized_columns = {str(column).lower() for column in renamed.columns}
        if not {"close", "volume"}.issubset(normalized_columns):
            return None
        return renamed[["close", "volume"]].copy()
