from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
import time


@dataclass
class EmotionSample:
    t: float
    valence: float
    arousal: float


class EmotionHistory:
    """Rolling buffer for valence/arousal with smoothing."""

    def __init__(self, maxlen: int = 200):
        self.buffer: Deque[EmotionSample] = deque(maxlen=maxlen)

    def add(self, valence: float, arousal: float, timestamp: Optional[float] = None) -> None:
        ts = timestamp if timestamp is not None else time.time()
        self.buffer.append(EmotionSample(ts, valence, arousal))

    def _last_n(self, n: int) -> Tuple[list[float], list[float]]:
        if n <= 0:
            return [], []
        vals = list(self.buffer)[-n:]
        return [v.valence for v in vals], [v.arousal for v in vals]

    def _avg_list(self, xs: list[float]) -> float:
        if not xs:
            return 0.0
        return sum(xs) / len(xs)

    def smoothed(self, window: int = 20) -> Dict[str, float]:
        v_list, a_list = self._last_n(window)
        return {
            "valence": self._avg_list(v_list),
            "arousal": self._avg_list(a_list),
        }

    def trend(self, window: int = 10, threshold: float = 0.05) -> str:
        vals, _ = self._last_n(window)
        if len(vals) < 2:
            return "stable"
        delta = vals[-1] - vals[0]
        if delta > threshold:
            return "rising"
        if delta < -threshold:
            return "falling"
        return "stable"

    def latest(self) -> Optional[EmotionSample]:
        return self.buffer[-1] if self.buffer else None


__all__ = ["EmotionHistory", "EmotionSample"]
