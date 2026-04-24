"""Risk management helpers for position safety checks.

This module keeps all risk logic in one place so trading decisions and
safety limits stay clearly separated.
"""

from config import DAILY_LOSS_LIMIT, PROFIT_TARGET, TRAILING_DRAWDOWN_LIMIT


def can_trade(day_pnl: float, trailing_pnl: float) -> bool:
    """Return True if risk limits still allow new trades."""
    if day_pnl <= DAILY_LOSS_LIMIT:
        return False
    if trailing_pnl <= TRAILING_DRAWDOWN_LIMIT:
        return False
    if day_pnl >= PROFIT_TARGET:
        return False
    return True
