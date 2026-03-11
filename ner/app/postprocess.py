"""Convert BIO-tagged NER predictions to structured JSON per domain.

Also contains time-word-to-HH:MM conversion for the uren domain.
The time expressions here mirror those in training/templates/uren_templates.json
(inverted: expression → HH:MM instead of HH:MM → expressions).
"""

# Reverse lookup: Dutch time words → HH:MM
_TIME_WORD_TO_HHMM: dict[str, str] = {
    "zes": "06:00", "half zeven": "06:30",
    "zeven": "07:00", "half acht": "07:30",
    "acht": "08:00", "half negen": "08:30",
    "negen": "09:00", "half tien": "09:30",
    "tien": "10:00", "half elf": "10:30",
    "elf": "11:00", "half twaalf": "11:30",
    "twaalf": "12:00", "half een": "12:30",
    "een": "13:00", "dertien": "13:00",
    "twee": "14:00", "veertien": "14:00", "half drie": "14:30",
    "drie": "15:00", "vijftien": "15:00", "half vier": "15:30",
    "vier": "16:00", "zestien": "16:00", "half vijf": "16:30",
    "vijf": "17:00", "zeventien": "17:00", "half zes": "17:30",
    "zes uur": "06:00", "zeven uur": "07:00", "acht uur": "08:00",
    "negen uur": "09:00", "tien uur": "10:00", "elf uur": "11:00",
    "twaalf uur": "12:00", "een uur": "13:00", "twee uur": "14:00",
    "drie uur": "15:00", "vier uur": "16:00", "vijf uur": "17:00",
    "kwart over acht": "08:15", "kwart voor negen": "08:45",
    "kwart over negen": "09:15", "kwart voor tien": "09:45",
    "kwart over twaalf": "12:15", "kwart voor een": "12:45",
    "kwart over vier": "16:15", "kwart voor vijf": "16:45",
}


def tokens_to_time(tokens: list[str]) -> str:
    """Convert extracted time tokens to HH:MM format."""
    raw = " ".join(tokens).strip().lower()

    if raw in _TIME_WORD_TO_HHMM:
        return _TIME_WORD_TO_HHMM[raw]

    # Try numeric: "8", "12", "17", "8 30"
    parts = raw.replace(":", " ").split()
    try:
        h = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 0
        if 1 <= h <= 7:  # assume afternoon in work context
            h += 12
        return f"{h:02d}:{m:02d}"
    except (ValueError, IndexError):
        pass

    return raw


def compute_duration(start: str, end: str) -> int:
    """Compute duration in minutes between two HH:MM times."""
    try:
        sh, sm = map(int, start.split(":"))
        eh, em = map(int, end.split(":"))
        return (eh * 60 + em) - (sh * 60 + sm)
    except (ValueError, AttributeError):
        return 0


def group_entities(
    predictions: list[tuple[str, str]],
) -> list[tuple[str, list[str]]]:
    """Group consecutive B-/I- tagged tokens into (entity_type, [tokens])."""
    groups: list[tuple[str, list[str]]] = []
    current_type: str | None = None
    current_tokens: list[str] = []

    for token, label in predictions:
        if label.startswith("B-"):
            if current_type is not None:
                groups.append((current_type, current_tokens))
            current_type = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_type == label[2:]:
            current_tokens.append(token)
        else:
            if current_type is not None:
                groups.append((current_type, current_tokens))
                current_type = None
                current_tokens = []

    if current_type is not None:
        groups.append((current_type, current_tokens))

    return groups


def bio_to_uren_entries(predictions: list[tuple[str, str]]) -> list[dict]:
    """Convert BIO-tagged predictions to structured uren entries."""
    entities = group_entities(predictions)
    entries = []
    current: dict[str, list[str]] = {}

    for etype, tokens in entities:
        if etype == "TIME_START":
            if current.get("TIME_START") and current.get("TIME_END"):
                entries.append(_build_uren_entry(current))
                current = {}
            current["TIME_START"] = tokens
        elif etype == "TIME_END":
            current["TIME_END"] = tokens
        elif etype == "DESC":
            current["DESC"] = tokens

    if current.get("TIME_START") and current.get("TIME_END"):
        entries.append(_build_uren_entry(current))

    return entries


def _build_uren_entry(current: dict[str, list[str]]) -> dict:
    start = tokens_to_time(current.get("TIME_START", []))
    end = tokens_to_time(current.get("TIME_END", []))
    desc = " ".join(current.get("DESC", [])).strip()
    return {
        "start": start,
        "eind": end,
        "omschrijving": desc.capitalize() if desc else "",
        "duur_minuten": compute_duration(start, end),
    }


def bio_to_km_entries(predictions: list[tuple[str, str]]) -> list[dict]:
    """Convert BIO-tagged predictions to structured km entries."""
    entities = group_entities(predictions)
    entries = []
    current: dict[str, list[str]] = {}

    for etype, tokens in entities:
        if etype == "LOC_FROM":
            if current.get("LOC_FROM") and current.get("LOC_TO"):
                entries.append(_build_km_entry(current))
                current = {}
            current["LOC_FROM"] = tokens
        elif etype == "LOC_TO":
            current["LOC_TO"] = tokens
        elif etype == "PURPOSE":
            current["PURPOSE"] = tokens

    if current.get("LOC_FROM") and current.get("LOC_TO"):
        entries.append(_build_km_entry(current))

    return entries


def _build_km_entry(current: dict[str, list[str]]) -> dict:
    return {
        "van": " ".join(current.get("LOC_FROM", [])).strip().capitalize(),
        "naar": " ".join(current.get("LOC_TO", [])).strip().capitalize(),
        "doel": " ".join(current.get("PURPOSE", [])).strip().capitalize(),
    }
