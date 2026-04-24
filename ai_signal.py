"""AI signal engine for directional futures decisions.

Gradient Boosting is an ensemble method that builds many small decision trees
one after another, where each new tree focuses on fixing mistakes made by the
previous trees. We use it instead of a simpler linear model because market
relationships are often non-linear (indicator combinations matter), and
GradientBoostingClassifier can capture those interactions while still being
faster and easier to operate than deep learning for this project size.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

TRAINING_LOOKBACK_DAYS = 252
RETRAIN_AFTER_DAYS = 30
CONFIDENCE_THRESHOLD = 0.60


# This function chooses a model file path that works in local and cloud runtimes.
def _resolve_default_model_path() -> Path:
    """Return a safe default model path in both local and cloud runtimes."""
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # Some hosted runners execute modules where __file__ is unavailable.
        base_dir = Path.cwd()
    return base_dir / "models" / "gradient_boosting_signal.pkl"


MODEL_PATH = _resolve_default_model_path()


class AISignalEngine:
    # This function creates the AI signal engine and restores a saved model if one exists.
    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.model = GradientBoostingClassifier(random_state=42)
        self.last_trained_at: Optional[datetime] = None
        self.is_trained = False
        self.last_confidence = 0.0
        self._load_model()

    # This function trains the Gradient Boosting model on the most recent 252 trading days.
    def train(self, market_data: pd.DataFrame) -> None:
        prepared_data = self._prepare_market_data(market_data)
        features = self._build_features(prepared_data)
        labels = self._build_labels(prepared_data)

        training_frame = (
            features.join(labels.rename("target"), how="inner")
            .dropna()
            .tail(TRAINING_LOOKBACK_DAYS)
        )
        if training_frame.empty:
            raise ValueError("Not enough valid rows to train the AI model.")

        x_train = training_frame[features.columns]
        y_train = training_frame["target"].astype(int)
        if y_train.nunique() < 2:
            raise ValueError("Training data must include both long and short examples.")

        self.model.fit(x_train, y_train)
        self.last_trained_at = datetime.now(timezone.utc)
        self.is_trained = True
        self._save_model()

    # This function predicts long, short, or flat and only trades when confidence is above 60%.
    def predict(self, market_data: pd.DataFrame) -> int:
        signal, _ = self.predict_with_confidence(market_data)
        return signal

    # This function predicts long, short, or flat and also returns the confidence score.
    def predict_with_confidence(self, market_data: pd.DataFrame) -> tuple[int, float]:
        if not self.is_trained:
            self.last_confidence = 0.0
            return 0, 0.0

        prepared_data = self._prepare_market_data(market_data)
        features = self._build_features(prepared_data).dropna()
        if features.empty:
            self.last_confidence = 0.0
            return 0, 0.0

        latest_features = features.iloc[[-1]]
        probabilities = self.model.predict_proba(latest_features)[0]
        classes = self.model.classes_

        best_index = int(np.argmax(probabilities))
        best_confidence = float(probabilities[best_index])
        best_class = int(classes[best_index])
        self.last_confidence = best_confidence

        if best_confidence <= CONFIDENCE_THRESHOLD:
            return 0, best_confidence
        signal = 1 if best_class > 0 else -1
        return signal, best_confidence

    # This function tells us if the model is older than 30 days and should be retrained.
    def retrain_needed(self, now: Optional[datetime] = None) -> bool:
        if not self.is_trained or self.last_trained_at is None:
            return True

        current_time = now if now else datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        return current_time - self.last_trained_at >= timedelta(days=RETRAIN_AFTER_DAYS)

    # This function standardizes column names and validates required market fields.
    def _prepare_market_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        if market_data is None or market_data.empty:
            raise ValueError("Market data is empty; cannot compute AI features.")

        data = market_data.copy()
        rename_map = {}
        if "Close" in data.columns and "close" not in data.columns:
            rename_map["Close"] = "close"
        if "Volume" in data.columns and "volume" not in data.columns:
            rename_map["Volume"] = "volume"
        data = data.rename(columns=rename_map)

        required_columns = {"close", "volume"}
        missing = required_columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns for AI features: {sorted(missing)}")

        data = data.sort_index()
        data["close"] = pd.to_numeric(data["close"], errors="coerce")
        data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
        return data.dropna(subset=["close", "volume"])

    # This function builds all required ML features from price and volume history.
    def _build_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        close = market_data["close"]
        volume = market_data["volume"]

        returns = close.pct_change()
        rsi_14 = self._calculate_rsi(close, period=14)
        ma_diff = close.rolling(20).mean() - close.rolling(50).mean()
        momentum_5 = close.pct_change(5)
        volatility_20 = returns.rolling(20).std()
        volume_relative_20 = volume / volume.rolling(20).mean()

        return pd.DataFrame(
            {
                "rsi_14": rsi_14,
                "ma_20_50_diff": ma_diff,
                "momentum_5": momentum_5,
                "volatility_20": volatility_20,
                "volume_relative_20": volume_relative_20,
            },
            index=market_data.index,
        )

    # This function builds labels for supervised learning using the next bar return direction.
    def _build_labels(self, market_data: pd.DataFrame) -> pd.Series:
        next_bar_return = market_data["close"].pct_change().shift(-1)
        labels = pd.Series(
            np.where(next_bar_return > 0, 1, -1),
            index=market_data.index,
            dtype="float64",
        )
        return labels.where(next_bar_return.notna())

    # This function calculates RSI so the model can detect overbought and oversold conditions.
    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        relative_strength = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + relative_strength))

    # This function saves the trained model and metadata to disk so restarts keep the AI state.
    def _save_model(self) -> None:
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": self.model,
                "last_trained_at": self.last_trained_at,
                "is_trained": self.is_trained,
            }
            with self.model_path.open("wb") as model_file:
                pickle.dump(payload, model_file)
        except OSError:
            # If the runtime file system is read-only, skip persistence without crashing.
            return

    # This function loads a previously saved model from disk when the bot starts up.
    def _load_model(self) -> None:
        if not self.model_path.exists():
            return

        try:
            with self.model_path.open("rb") as model_file:
                payload = pickle.load(model_file)
        except (OSError, EOFError, pickle.UnpicklingError):
            return

        if not isinstance(payload, dict):
            return

        model = payload.get("model")
        if isinstance(model, GradientBoostingClassifier):
            self.model = model
            self.is_trained = bool(payload.get("is_trained", True))

            trained_at = payload.get("last_trained_at")
            if isinstance(trained_at, datetime):
                if trained_at.tzinfo is None:
                    trained_at = trained_at.replace(tzinfo=timezone.utc)
                self.last_trained_at = trained_at


default_engine: Optional[AISignalEngine] = None


# This function keeps backward compatibility with the existing bot scaffold API.
def generate_signal(market_snapshot: Dict[str, Any]) -> int:
    global default_engine
    if default_engine is None:
        default_engine = AISignalEngine()

    history = market_snapshot.get("history")
    if not isinstance(history, pd.DataFrame):
        return 0

    try:
        if default_engine.retrain_needed():
            default_engine.train(history)
        return default_engine.predict(history)
    except ValueError:
        return 0
