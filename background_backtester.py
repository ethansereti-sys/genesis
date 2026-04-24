"""Background backtesting module for continuous strategy improvement.

A backtest is a replay of past market data that estimates how a strategy would
have performed if it had traded in that historical period. Running backtests
continuously in the background helps the bot learn whether small parameter
changes are improving or hurting performance, so it can adapt over time
without blocking live trading logic.
"""

from __future__ import annotations

import json
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from AlgorithmImports import Resolution

from config import ACCOUNT_SIZE, DAILY_LOSS_LIMIT, TRAILING_DRAWDOWN_LIMIT


# This function returns a reliable file path for saving background backtest logs.
def _resolve_log_path() -> Path:
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    return base_dir / "backtest_log.json"


@dataclass
class BacktestResult:
    timestamp: str
    parameters: Dict[str, float]
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    score: float
    worst_daily_pnl: float


class BackgroundBacktester:
    """Runs background optimization cycles in a daemon thread."""

    # This function sets up all background backtester state and callbacks.
    def __init__(
        self,
        algorithm,
        es_symbol,
        nq_symbol,
        get_live_parameters: Optional[Callable[[], Dict[str, float]]] = None,
        apply_live_parameters: Optional[Callable[[Dict[str, float]], None]] = None,
    ) -> None:
        self.algorithm = algorithm
        self.symbols = {"ES": es_symbol, "NQ": nq_symbol}

        self.get_live_parameters = get_live_parameters or self._default_get_live_parameters
        self.apply_live_parameters = apply_live_parameters or self._default_apply_live_parameters

        self.log_path = _resolve_log_path()
        self.cycle_interval = timedelta(hours=6)
        self.history_lookback = timedelta(days=60)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_cycle_requested = True

        self.cycles_run = 0
        self.last_backtest_time: Optional[datetime] = None
        self.current_best_score = float("-inf")
        self.current_live_parameters = self._default_get_live_parameters()

        self._candidate_tracker: Dict[str, Dict[str, Any]] = {}
        self._pending_upgrade_key: Optional[str] = None

        self._last_non_flat_time: Optional[datetime] = None

    # This function starts the daemon background thread that runs backtest cycles.
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_forever,
                name="background-backtester",
                daemon=True,
            )
            self._thread.start()
            self.algorithm.Debug("[BG-BACKTEST] Background backtester thread started.")

    # This function stops the background loop cleanly when requested.
    def stop(self, join_timeout_seconds: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_seconds)

    # This function returns key health/status fields for monitoring.
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "last_backtest_time": (
                    self.last_backtest_time.isoformat() if self.last_backtest_time else None
                ),
                "current_best_score": (
                    None
                    if self.current_best_score == float("-inf")
                    else round(float(self.current_best_score), 6)
                ),
                "upgrade_pending": self._pending_upgrade_key is not None,
                "cycles_run": self.cycles_run,
            }

    # This function runs one manual cycle immediately without waiting 6 hours.
    def run_cycle_now(self) -> None:
        self._run_cycle()

    # This function executes one queued cycle request from the main algorithm thread.
    def process_pending_cycle(self) -> bool:
        with self._lock:
            should_run = self._pending_cycle_requested
            if should_run:
                self._pending_cycle_requested = False
        if not should_run:
            return False
        self._run_cycle()
        return True

    # This function provides default live parameters when no callback is supplied.
    def _default_get_live_parameters(self) -> Dict[str, float]:
        return {
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "confidence_threshold": 0.60,
            "stop_loss_points": 10.0,
            "take_profit_points": 20.0,
        }

    # This function stores promoted parameters locally when no apply callback is supplied.
    def _default_apply_live_parameters(self, parameters: Dict[str, float]) -> None:
        with self._lock:
            self.current_live_parameters = dict(parameters)

    # This function runs a continuous 6-hour loop in a daemon thread.
    def _run_forever(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                self._pending_cycle_requested = True
            sleep_seconds = max(0.0, self.cycle_interval.total_seconds())
            self._stop_event.wait(timeout=sleep_seconds)

    # This function executes one full cycle: fetch data, test variations, and evaluate upgrades.
    def _run_cycle(self) -> None:
        cycle_id = None
        try:
            with self._lock:
                self.cycles_run += 1
                cycle_id = self.cycles_run

            historical_data = self._fetch_historical_data()
            if not historical_data:
                self.algorithm.Debug("[BG-BACKTEST] No history returned; skipping cycle.")
                return

            live_parameters = self.get_live_parameters()
            baseline_result = self._run_full_backtest(historical_data, live_parameters)
            results_to_log = [baseline_result]

            variations = self._generate_variations(live_parameters, count=5)
            best_candidate: Optional[Tuple[BacktestResult, float]] = None

            for variation in variations:
                result = self._run_full_backtest(historical_data, variation)
                results_to_log.append(result)

                improvement = self._score_improvement(baseline_result.score, result.score)
                if improvement > 0.10:
                    if best_candidate is None or improvement > best_candidate[1]:
                        best_candidate = (result, improvement)

            self._append_results_to_log(results_to_log)
            self._mark_cycle_complete(baseline_result)

            if best_candidate is not None and cycle_id is not None:
                self._register_candidate(best_candidate[0], best_candidate[1], cycle_id)

            self._attempt_promotion_if_safe()
        except Exception as error:
            self.algorithm.Debug(f"[BG-BACKTEST] Cycle failed safely: {error}")

    # This function fetches and normalizes the last 60 days of OHLCV data for ES and NQ.
    def _fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        data_by_ticker: Dict[str, pd.DataFrame] = {}
        for ticker, symbol in self.symbols.items():
            history = self.algorithm.History(symbol, self.history_lookback, Resolution.Minute)
            frame = self._normalize_history_frame(history, symbol)
            if frame is None or frame.empty:
                continue

            bars_5m = (
                frame[["open", "high", "low", "close", "volume"]]
                .resample("5min")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            if not bars_5m.empty:
                data_by_ticker[ticker] = bars_5m

        return data_by_ticker

    # This function normalizes Lean history output into lowercase OHLCV columns.
    def _normalize_history_frame(self, history: pd.DataFrame, symbol) -> Optional[pd.DataFrame]:
        if history is None or history.empty:
            return None

        frame = history
        if "symbol" in getattr(frame.index, "names", []):
            frame = frame.xs(symbol, level="symbol")

        renamed = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        needed = {"open", "high", "low", "close", "volume"}
        if not needed.issubset(set(map(str.lower, renamed.columns))):
            normalized = {}
            for column in renamed.columns:
                normalized[str(column).lower()] = column
            if not needed.issubset(normalized):
                return None
            renamed = renamed[[normalized[name] for name in ("open", "high", "low", "close", "volume")]]
            renamed.columns = ["open", "high", "low", "close", "volume"]
        else:
            renamed = renamed[["open", "high", "low", "close", "volume"]]

        for column in ("open", "high", "low", "close", "volume"):
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
        return renamed.dropna()

    # This function creates 5 small random parameter variations around current live settings.
    def _generate_variations(self, current: Dict[str, float], count: int) -> List[Dict[str, float]]:
        variations: List[Dict[str, float]] = []
        for _ in range(count):
            shift = random.randint(-3, 3)

            rsi_oversold = max(10.0, min(45.0, current["rsi_oversold"] + shift))
            rsi_overbought = max(55.0, min(90.0, current["rsi_overbought"] + shift))
            if rsi_oversold > rsi_overbought - 5.0:
                rsi_oversold = rsi_overbought - 5.0

            variation = {
                "rsi_oversold": round(rsi_oversold, 2),
                "rsi_overbought": round(rsi_overbought, 2),
                "confidence_threshold": round(
                    max(0.45, min(0.95, current["confidence_threshold"] + random.uniform(-0.05, 0.05))), 4
                ),
                "stop_loss_points": round(
                    max(1.0, current["stop_loss_points"] + random.uniform(-2.0, 2.0)), 3
                ),
                "take_profit_points": round(
                    max(1.0, current["take_profit_points"] + random.uniform(-5.0, 5.0)), 3
                ),
            }
            variations.append(variation)
        return variations

    # This function runs a full strategy backtest over ES/NQ data for one parameter set.
    def _run_full_backtest(
        self, historical_data: Dict[str, pd.DataFrame], parameters: Dict[str, float]
    ) -> BacktestResult:
        trade_records: List[Dict[str, Any]] = []

        for ticker, frame in historical_data.items():
            ticker_trades = self._simulate_symbol_trades(ticker, frame, parameters)
            trade_records.extend(ticker_trades)

        trade_records.sort(key=lambda item: item["time"])

        realized_pnls = [record["pnl"] for record in trade_records]
        wins = sum(1 for pnl in realized_pnls if pnl > 0.0)
        losses = [pnl for pnl in realized_pnls if pnl < 0.0]
        gains = [pnl for pnl in realized_pnls if pnl > 0.0]

        total_trades = len(realized_pnls)
        win_rate = (wins / total_trades) if total_trades else 0.0
        profit_factor = (
            (sum(gains) / abs(sum(losses)))
            if losses
            else (float(sum(gains)) if gains else 0.0)
        )

        returns = np.array(realized_pnls, dtype=float) / float(max(ACCOUNT_SIZE, 1))
        sharpe_ratio = self._compute_sharpe(returns)

        equity_curve = np.cumsum(np.array(realized_pnls, dtype=float))
        max_drawdown = self._compute_max_drawdown(equity_curve)
        worst_daily_pnl = self._compute_worst_daily_pnl(trade_records)

        score = (win_rate * 0.4) + (sharpe_ratio * 0.4) + (profit_factor * 0.2)
        timestamp = datetime.now(timezone.utc).isoformat()

        return BacktestResult(
            timestamp=timestamp,
            parameters=dict(parameters),
            win_rate=float(win_rate),
            sharpe_ratio=float(sharpe_ratio),
            max_drawdown=float(max_drawdown),
            profit_factor=float(profit_factor),
            score=float(score),
            worst_daily_pnl=float(worst_daily_pnl),
        )

    # This function simulates trades for one symbol using stop loss/take profit and signal filters.
    def _simulate_symbol_trades(
        self, ticker: str, frame: pd.DataFrame, parameters: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        if len(frame) < 80:
            return []

        enriched = frame.copy()
        enriched["rsi_14"] = self._calculate_rsi(enriched["close"], period=14)
        enriched["ma_diff"] = enriched["close"].rolling(20).mean() - enriched["close"].rolling(50).mean()
        enriched["momentum_5"] = enriched["close"].pct_change(5)
        enriched["returns"] = enriched["close"].pct_change()
        enriched["volatility_20"] = enriched["returns"].rolling(20).std()
        enriched["volume_relative_20"] = enriched["volume"] / enriched["volume"].rolling(20).mean()
        enriched = enriched.dropna()

        if enriched.empty:
            return []

        multiplier = 50.0 if ticker == "ES" else 20.0
        position = 0
        entry_price = 0.0
        stop_price = 0.0
        target_price = 0.0
        trades: List[Dict[str, Any]] = []

        for timestamp, row in enriched.iterrows():
            signal, confidence = self._generate_signal_from_row(row, parameters)
            close_price = float(row["close"])
            high_price = float(row["high"])
            low_price = float(row["low"])

            if position == 1:
                if low_price <= stop_price:
                    pnl = (stop_price - entry_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0
                elif high_price >= target_price:
                    pnl = (target_price - entry_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0
                elif signal <= 0:
                    pnl = (close_price - entry_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0

            elif position == -1:
                if high_price >= stop_price:
                    pnl = (entry_price - stop_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0
                elif low_price <= target_price:
                    pnl = (entry_price - target_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0
                elif signal >= 0:
                    pnl = (entry_price - close_price) * multiplier
                    trades.append({"time": timestamp, "pnl": pnl, "confidence": confidence})
                    position = 0

            if position == 0 and signal != 0:
                position = signal
                entry_price = close_price
                if signal > 0:
                    stop_price = entry_price - parameters["stop_loss_points"]
                    target_price = entry_price + parameters["take_profit_points"]
                else:
                    stop_price = entry_price + parameters["stop_loss_points"]
                    target_price = entry_price - parameters["take_profit_points"]

        return trades

    # This function creates a directional signal and confidence score from one feature row.
    def _generate_signal_from_row(self, row: pd.Series, parameters: Dict[str, float]) -> Tuple[int, float]:
        ma_component = np.tanh(abs(float(row["ma_diff"]) / max(float(row["close"]), 1e-6)) * 250.0)
        momentum_component = np.tanh(abs(float(row["momentum_5"])) * 30.0)
        rsi_component = np.tanh(abs((float(row["rsi_14"]) - 50.0) / 15.0))
        volume_component = np.tanh(abs(float(row["volume_relative_20"]) - 1.0) * 2.0)
        volatility_component = 1.0 - np.tanh(float(row["volatility_20"]) * 50.0)

        confidence = (
            (ma_component * 0.30)
            + (momentum_component * 0.25)
            + (rsi_component * 0.20)
            + (volume_component * 0.15)
            + (volatility_component * 0.10)
        )
        confidence = float(max(0.0, min(1.0, confidence)))

        if confidence < parameters["confidence_threshold"]:
            return 0, confidence

        rsi = float(row["rsi_14"])
        ma_diff = float(row["ma_diff"])
        momentum = float(row["momentum_5"])

        if rsi <= parameters["rsi_oversold"] and ma_diff > 0.0 and momentum > 0.0:
            return 1, confidence
        if rsi >= parameters["rsi_overbought"] and ma_diff < 0.0 and momentum < 0.0:
            return -1, confidence
        return 0, confidence

    # This function computes RSI for backtest signal generation.
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    # This function computes a simple Sharpe ratio from trade-level returns.
    def _compute_sharpe(self, returns: np.ndarray) -> float:
        if returns.size < 2:
            return 0.0
        std = float(np.std(returns, ddof=1))
        if std == 0.0:
            return 0.0
        return float(np.sqrt(252.0) * np.mean(returns) / std)

    # This function computes max drawdown as current equity minus peak equity.
    def _compute_max_drawdown(self, equity_curve: np.ndarray) -> float:
        if equity_curve.size == 0:
            return 0.0
        peak = float("-inf")
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, float(value))
            max_drawdown = min(max_drawdown, float(value) - peak)
        return float(max_drawdown)

    # This function computes the worst single-day realized P&L from all simulated trades.
    def _compute_worst_daily_pnl(self, trade_records: List[Dict[str, Any]]) -> float:
        if not trade_records:
            return 0.0
        daily: Dict[str, float] = {}
        for record in trade_records:
            date_key = pd.Timestamp(record["time"]).date().isoformat()
            daily[date_key] = daily.get(date_key, 0.0) + float(record["pnl"])
        return float(min(daily.values()))

    # This function measures percentage score improvement versus the current live score.
    def _score_improvement(self, baseline_score: float, candidate_score: float) -> float:
        denominator = abs(baseline_score) if baseline_score != 0.0 else 1.0
        return (candidate_score - baseline_score) / denominator

    # This function records cycle-level metadata after a baseline backtest completes.
    def _mark_cycle_complete(self, baseline_result: BacktestResult) -> None:
        with self._lock:
            self.last_backtest_time = datetime.now(timezone.utc)
            self.current_best_score = max(self.current_best_score, baseline_result.score)

    # This function stores a candidate parameter set and tracks how many cycles it has won.
    def _register_candidate(
        self, result: BacktestResult, improvement: float, cycle_id: int
    ) -> None:
        signature = self._parameter_signature(result.parameters)
        with self._lock:
            tracker = self._candidate_tracker.setdefault(
                signature,
                {
                    "parameters": result.parameters,
                    "cycles": set(),
                    "best_improvement": 0.0,
                    "latest_result": result,
                },
            )
            tracker["cycles"].add(cycle_id)
            tracker["best_improvement"] = max(float(tracker["best_improvement"]), float(improvement))
            tracker["latest_result"] = result

            if len(tracker["cycles"]) >= 3:
                self._pending_upgrade_key = signature

    # This function decides whether a pending candidate can be safely promoted to live settings.
    def _attempt_promotion_if_safe(self) -> None:
        with self._lock:
            pending_key = self._pending_upgrade_key
            if pending_key is None:
                return
            tracker = self._candidate_tracker.get(pending_key)
            if tracker is None:
                self._pending_upgrade_key = None
                return
            result: BacktestResult = tracker["latest_result"]
            if len(tracker["cycles"]) < 3:
                return
            new_params = dict(tracker["parameters"])
            improvement = float(tracker["best_improvement"])

        if not self._passes_prop_risk_limits(result):
            return
        if not self._is_flat_for_minimum_minutes(10):
            return

        old_params = self.get_live_parameters()

        try:
            self.apply_live_parameters(new_params)
            with self._lock:
                self.current_live_parameters = dict(new_params)
                self._pending_upgrade_key = None
                self._candidate_tracker.pop(self._parameter_signature(new_params), None)

            self.algorithm.Debug(
                "[BG-BACKTEST] Auto-promoted parameters at "
                f"{datetime.now(timezone.utc).isoformat()} | "
                f"old={old_params} | new={new_params} | "
                f"improvement={improvement:.4f}"
            )
        except Exception as error:
            self.algorithm.Debug(f"[BG-BACKTEST] Promotion failed safely: {error}")

    # This function checks prop-firm risk rules before allowing any parameter promotion.
    def _passes_prop_risk_limits(self, result: BacktestResult) -> bool:
        if result.worst_daily_pnl <= DAILY_LOSS_LIMIT:
            return False
        if result.max_drawdown <= TRAILING_DRAWDOWN_LIMIT:
            return False
        return True

    # This function verifies the live bot has stayed flat for at least a minimum number of minutes.
    def _is_flat_for_minimum_minutes(self, minutes: int) -> bool:
        now = datetime.now(timezone.utc)
        is_invested = bool(self.algorithm.Portfolio.Invested)

        if is_invested:
            self._last_non_flat_time = now
            return False

        if self._last_non_flat_time is None:
            self._last_non_flat_time = now - timedelta(minutes=minutes)
            return True

        return (now - self._last_non_flat_time) >= timedelta(minutes=minutes)

    # This function creates a stable text key for identifying matching parameter sets.
    def _parameter_signature(self, parameters: Dict[str, float]) -> str:
        ordered = {
            "rsi_oversold": round(float(parameters["rsi_oversold"]), 2),
            "rsi_overbought": round(float(parameters["rsi_overbought"]), 2),
            "confidence_threshold": round(float(parameters["confidence_threshold"]), 4),
            "stop_loss_points": round(float(parameters["stop_loss_points"]), 3),
            "take_profit_points": round(float(parameters["take_profit_points"]), 3),
        }
        return json.dumps(ordered, sort_keys=True)

    # This function appends backtest results to disk and keeps only the latest 200 entries.
    def _append_results_to_log(self, results: List[BacktestResult]) -> None:
        records = [self._result_to_log_dict(result) for result in results]
        existing: List[Dict[str, Any]] = []

        try:
            if self.log_path.exists():
                with self.log_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, list):
                        existing = loaded
        except (OSError, json.JSONDecodeError):
            existing = []

        combined = (existing + records)[-200:]
        try:
            with self.log_path.open("w", encoding="utf-8") as handle:
                json.dump(combined, handle, indent=2)
        except OSError as error:
            self.algorithm.Debug(f"[BG-BACKTEST] Could not write backtest_log.json: {error}")

    # This function converts one backtest result to the exact JSON shape used in the log file.
    def _result_to_log_dict(self, result: BacktestResult) -> Dict[str, Any]:
        return {
            "timestamp": result.timestamp,
            "parameters_tested": result.parameters,
            "win_rate": result.win_rate,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "profit_factor": result.profit_factor,
            "score": result.score,
        }
