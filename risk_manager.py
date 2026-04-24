"""Centralized risk controls for the futures bot."""

from datetime import timedelta

from config import (
    DAILY_LOSS_LIMIT,
    EOD_FLATTEN_HOUR,
    EOD_FLATTEN_MINUTE,
    TRAILING_DRAWDOWN_LIMIT,
    WATCHDOG_NO_SIGNAL_MINUTES,
)


class RiskManager:
    # This function creates a risk manager and starts tracking equity state.
    def __init__(self, algorithm, tracked_symbol):
        self.algorithm = algorithm
        self.tracked_symbol = tracked_symbol

        current_equity = float(self.algorithm.Portfolio.TotalPortfolioValue)
        self.current_trading_day = self.algorithm.Time.date()
        self.day_start_equity = current_equity
        self.daily_pnl = 0.0

        self.peak_equity = current_equity
        self.trailing_drawdown = 0.0

        self.last_signal_time = None
        self.watchdog_start_time = None

    # This function records the exact time when a new trade signal is fired.
    def record_signal_fired(self):
        self.last_signal_time = self.algorithm.Time
        self.watchdog_start_time = self.algorithm.Time

    # This function checks all risk rules and closes positions when needed.
    def check_risk(self) -> bool:
        self._reset_for_new_trading_day_if_needed()
        current_equity = self._update_equity_tracking()

        if self.daily_pnl <= DAILY_LOSS_LIMIT:
            self._flatten_with_log(
                "[RISK] Daily loss limit triggered. "
                f"Daily P&L {self.daily_pnl:.2f} <= limit {DAILY_LOSS_LIMIT:.2f}. "
                "Liquidating all positions."
            )
            return False

        if self.trailing_drawdown <= TRAILING_DRAWDOWN_LIMIT:
            self._flatten_with_log(
                "[RISK] Trailing drawdown limit triggered. "
                f"Drawdown {self.trailing_drawdown:.2f} <= limit {TRAILING_DRAWDOWN_LIMIT:.2f}. "
                f"Current equity {current_equity:.2f}, peak equity {self.peak_equity:.2f}. "
                "Liquidating all positions."
            )
            return False

        current_time = self.algorithm.Time
        eod_cutoff = current_time.replace(
            hour=EOD_FLATTEN_HOUR,
            minute=EOD_FLATTEN_MINUTE,
            second=0,
            microsecond=0,
        )
        if current_time >= eod_cutoff:
            self._flatten_with_log(
                "[RISK] End-of-day flatten triggered. "
                f"Current time {current_time.strftime('%H:%M:%S')} is at/after "
                f"{EOD_FLATTEN_HOUR:02d}:{EOD_FLATTEN_MINUTE:02d}. "
                "Liquidating all positions."
            )
            return False

        if self._watchdog_should_flatten():
            minutes_since_signal = self._minutes_since_last_signal()
            self._flatten_with_log(
                "[RISK] No-signal watchdog triggered. "
                f"No trade signal for {minutes_since_signal:.1f} minutes during market hours "
                f"(limit: {WATCHDOG_NO_SIGNAL_MINUTES} minutes). "
                "Liquidating all positions."
            )
            return False

        return True

    # This function resets daily P&L tracking at the start of each new day.
    def _reset_for_new_trading_day_if_needed(self):
        current_day = self.algorithm.Time.date()
        if current_day == self.current_trading_day:
            return

        self.current_trading_day = current_day
        self.day_start_equity = float(self.algorithm.Portfolio.TotalPortfolioValue)
        self.daily_pnl = 0.0
        self.last_signal_time = None
        self.watchdog_start_time = None

        self.algorithm.Debug(
            "[RISK] New trading day detected. "
            f"Reset daily P&L baseline to {self.day_start_equity:.2f}."
        )

    # This function updates current equity, peak equity, and trailing drawdown.
    def _update_equity_tracking(self) -> float:
        current_equity = float(self.algorithm.Portfolio.TotalPortfolioValue)
        self.daily_pnl = current_equity - self.day_start_equity

        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        self.trailing_drawdown = current_equity - self.peak_equity
        return current_equity

    # This function determines whether the instrument is currently in market hours.
    def _is_market_open(self) -> bool:
        if not self.algorithm.Securities.ContainsKey(self.tracked_symbol):
            return False
        security = self.algorithm.Securities[self.tracked_symbol]
        return security.Exchange.Hours.IsOpen(self.algorithm.Time, False)

    # This function checks whether the no-signal watchdog should flatten positions.
    def _watchdog_should_flatten(self) -> bool:
        if not self._is_market_open():
            self.watchdog_start_time = None
            return False

        if self.last_signal_time is None and self.watchdog_start_time is None:
            self.watchdog_start_time = self.algorithm.Time
            return False

        return self._minutes_since_last_signal() >= WATCHDOG_NO_SIGNAL_MINUTES

    # This function calculates how many minutes have passed since the last signal.
    def _minutes_since_last_signal(self) -> float:
        reference_time = self.last_signal_time or self.watchdog_start_time
        if reference_time is None:
            return 0.0
        delta = self.algorithm.Time - reference_time
        return delta / timedelta(minutes=1)

    # This function logs the rule that triggered and liquidates all positions.
    def _flatten_with_log(self, message: str):
        self.algorithm.Debug(message)
        self.algorithm.Liquidate()
