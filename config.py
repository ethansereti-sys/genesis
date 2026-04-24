"""Centralized configuration for the futures trading bot.

Keep all easy-to-change settings in one place so you don't have to hunt
through strategy code to update risk or timing parameters.
"""

# Trading instrument symbol (E-mini S&P 500 futures)
INSTRUMENT = "ES"

# Account and risk settings
ACCOUNT_SIZE = 150000
DAILY_LOSS_LIMIT = -1500
TRAILING_DRAWDOWN_LIMIT = -3000
PROFIT_TARGET = 3000

# End-of-day position flatten time (24-hour clock)
EOD_FLATTEN_HOUR = 15
EOD_FLATTEN_MINUTE = 55

# Optional webhook for TraderPost integration (left blank for now)
TRADERSPOST_WEBHOOK_URL = ""

# Safety watchdog: flatten if no signal appears within this many minutes.
WATCHDOG_NO_SIGNAL_MINUTES = 60
