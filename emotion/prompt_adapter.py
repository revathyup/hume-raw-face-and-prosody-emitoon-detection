from __future__ import annotations
from typing import Dict


def build_emotion_context(valence: float, arousal: float, trend: str) -> str:
    """Create structured emotion context block for GPT prompt injection."""
    return (
        "=== USER EMOTION STATE (last 15 seconds) ===\n"
        f"Smoothed Valence: {valence:.2f}  (-1 = negative, +1 = positive)\n"
        f"Smoothed Arousal: {arousal:.2f}  (-1 = calm, +1 = excited)\n"
        f"Trend: {trend}\n\n"
        "Guidance:\n"
        "- If valence is low → be supportive, soft tone.\n"
        "- If arousal is high → respond clearly and calmly.\n"
        "- If valence is high → allow more playful tone.\n"
        "- If arousal is low → use more engaging language.\n"
    )


def describe_tone(valence: float, arousal: float) -> str:
    """Human-readable hint for console about tone adjustment."""
    tone = []
    if valence > 0.3:
        tone.append("warm")
    elif valence < -0.3:
        tone.append("supportive")
    else:
        tone.append("neutral")

    if arousal > 0.3:
        tone.append("calming")
    elif arousal < -0.3:
        tone.append("energizing")

    if not tone:
        return "balanced"
    return " & ".join(tone)


__all__ = ["build_emotion_context", "describe_tone"]
