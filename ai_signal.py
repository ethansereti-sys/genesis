"""Signal generation module.

This file is responsible for turning market data into a directional idea
(e.g., bullish, bearish, or neutral). For now, it returns a placeholder
signal so the project has a clear extension point for future AI logic.
"""

from typing import Dict


def generate_signal(market_snapshot: Dict) -> int:
    """Return a basic trading signal.

    Args:
        market_snapshot: A dictionary of market values (price, volume, etc.).

    Returns:
        int:
            1  -> bullish (go long)
           -1  -> bearish (go short)
            0  -> neutral (no trade)
    """
    # Placeholder logic for beginner-friendly scaffolding.
    # You can replace this with indicators, ML model output,
    # or any custom decision rules later.
    return 0
