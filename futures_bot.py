"""Main Lean algorithm scaffold for a futures trading bot.

This is the primary entry point where QuantConnect Lean will run the
strategy lifecycle methods.
"""

from AlgorithmImports import *

from ai_signal import generate_signal
from config import ACCOUNT_SIZE, EOD_FLATTEN_HOUR, EOD_FLATTEN_MINUTE, INSTRUMENT
from risk_manager import can_trade


class FuturesBot(QCAlgorithm):
    """Beginner-friendly structure for a futures strategy."""

    def Initialize(self) -> None:
        self.SetStartDate(2024, 1, 1)
        self.SetCash(ACCOUNT_SIZE)

        instrument_map = {
            "ES": Futures.Indices.SP500EMini,
        }
        if INSTRUMENT not in instrument_map:
            raise ValueError(f"Unsupported instrument in config: {INSTRUMENT}")

        # Add continuous futures contract selected in config.py.
        self.future = self.AddFuture(instrument_map[INSTRUMENT]).Symbol

        # Schedule end-of-day flattening to reduce overnight risk.
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(EOD_FLATTEN_HOUR, EOD_FLATTEN_MINUTE),
            self.FlattenPositions,
        )

    def OnData(self, data: Slice) -> None:
        if self.future not in data.Bars:
            return

        # Basic placeholders for risk state tracking.
        day_pnl = 0.0
        trailing_pnl = 0.0
        if not can_trade(day_pnl=day_pnl, trailing_pnl=trailing_pnl):
            return

        market_snapshot = {
            "price": data.Bars[self.future].Close,
        }
        signal = generate_signal(market_snapshot)

        if signal == 1 and not self.Portfolio[self.future].Invested:
            self.MarketOrder(self.future, 1)
        elif signal == -1 and self.Portfolio[self.future].Quantity >= 0:
            self.MarketOrder(self.future, -1)

    def FlattenPositions(self) -> None:
        self.Liquidate()
