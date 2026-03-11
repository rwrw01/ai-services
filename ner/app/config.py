import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR / "models")))

# Base model for fine-tuning (used by training scripts)
BASE_MODEL = "DTAI-KULeuven/robbertje-1-gb-non-shuffled"

# Domains and their label sets
DOMAIN_LABELS: dict[str, list[str]] = {
    "uren": [
        "O",
        "B-TIME_START", "I-TIME_START",
        "B-TIME_END", "I-TIME_END",
        "B-DESC", "I-DESC",
    ],
    "km": [
        "O",
        "B-LOC_FROM", "I-LOC_FROM",
        "B-LOC_TO", "I-LOC_TO",
        "B-PURPOSE", "I-PURPOSE",
    ],
}
