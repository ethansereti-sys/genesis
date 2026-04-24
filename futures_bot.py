"""Main Lean algorithm scaffold for a futures trading bot.

This is the primary entry point where QuantConnect Lean will run the
strategy lifecycle methods.
"""

from AlgorithmImports import *

from ai_signal import generate_signal
from config import ACCOUNT_SIZE, INSTRUMENT
from risk_manager import RiskManager


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

        # Hand off all safety checks to the dedicated risk manager.
        self.risk_manager = RiskManager(self, self.future)

    def OnData(self, data: Slice) -> None:
        if not self.risk_manager.check_risk():
            return

        if self.future not in data.Bars:
            return

        history = self._build_market_history()
        if history is None:
            return

        market_snapshot = {
            "price": data.Bars[self.future].Close,
            "history": history,
        }
        signal = generate_signal(market_snapshot)
        if signal in (1, -1):
            self.risk_manager.record_signal_fired()

        if signal == 1 and not self.Portfolio[self.future].Invested:
            self.MarketOrder(self.future, 1)
        elif signal == -1 and self.Portfolio[self.future].Quantity >= 0:
            self.MarketOrder(self.future, -1)

    def _build_market_history(self):
        # Pull enough daily bars so the model can compute indicators and train.
        history = self.History(self.future, 320, Resolution.Daily)
        if history.empty:
            return None

        if "symbol" in getattr(history.index, "names", []):
            history = history.xs(self.future, level="symbol")

        normalized_columns = {str(col).lower() for col in history.columns}
        expected_columns = {"close", "volume"}
        if not expected_columns.issubset(normalized_columns):
            return None
        selected = history.rename(
            columns={
                "Close": "close",
                "Volume": "volume",
            }
        )
        return selected[["close", "volume"]]
