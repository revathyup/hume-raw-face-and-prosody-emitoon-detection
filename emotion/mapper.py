from __future__ import annotations
from typing import Dict, Tuple
import time

# Simple weights via equal averaging within groups. Names normalized to lowercase.
# Hume emotion names expected to match these strings; we normalize by lower() stripping spaces.

_POSITIVE = {
    "joy",
    "amusement",
    "ecstasy",
    "love",
    "romance",
    "satisfaction",
    "contentment",
    "relief",
    "admiration",
    "adoration",
    "aesthetic appreciation",
    "awe",
    "enthusiasm",
    "excitement",
    "pride",
    "triumph",
}

_NEGATIVE = {
    "anger",
    "annoyance",
    "anxiety",
    "fear",
    "distress",
    "sadness",
    "pain",
    "horror",
    "disappointment",
    "disgust",
    "contempt",
    "shame",
    "embarrassment",
    "guilt",
    "envy",
}

_HIGH_AROUSAL = {
    "joy",
    "amusement",
    "excitement",
    "ecstasy",
    "anger",
    "fear",
    "anxiety",
    "surprise (positive)",
    "surprise (negative)",
    "horror",
    "distress",
    "triumph",
    "pride",
}

_LOW_AROUSAL = {
    "calmness",
    "contemplation",
    "contentment",
    "nostalgia",
    "tiredness",
    "boredom",
    "relief",
}


def _norm_name(name: str) -> str:
    return name.strip().lower()


def _avg(scores: Dict[str, float], keys: set[str]) -> float:
    vals = [scores[k] for k in keys if k in scores]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _clamp01(x: float) -> float:
    return max(min(x, 1.0), 0.0)


def _clamp_signed(x: float) -> float:
    return max(min(x, 1.0), -1.0)


def compute_va(raw_scores: Dict[str, float]) -> Dict[str, float]:
    """Compute valence/arousal from 48-d Hume face emotion scores.

    Args:
        raw_scores: mapping from emotion name (string) to score (0-1).

    Returns:
        dict with valence, arousal, timestamp, raw (original scores).
    """
    # Normalize keys
    scores = {_norm_name(k): _clamp01(v) for k, v in raw_scores.items()}

    pos = _avg(scores, _POSITIVE)
    neg = _avg(scores, _NEGATIVE)
    high = _avg(scores, _HIGH_AROUSAL)
    low = _avg(scores, _LOW_AROUSAL)

    valence = _clamp_signed(pos - neg)
    arousal = _clamp_signed(high - low)

    return {
        "valence": valence,
        "arousal": arousal,
        "timestamp": time.time(),
        "raw": raw_scores,
    }


__all__ = ["compute_va"]
