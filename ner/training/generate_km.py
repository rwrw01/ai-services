"""Generate synthetic NER training data for the 'km' domain.

Usage:
    python -m training.generate_km
"""

import random

from training.common import (
    add_stt_noise,
    load_templates,
    tagged_entity,
    tagged_o,
    write_splits,
)

B_LF, I_LF = "B-LOC_FROM", "I-LOC_FROM"
B_LT, I_LT = "B-LOC_TO", "I-LOC_TO"
B_P, I_P = "B-PURPOSE", "I-PURPOSE"

# Prepositions used in "building in city" compounds
_CITY_PREPS = ["in", ""]


def _pick_location(templates: dict, rng: random.Random) -> str:
    """Pick a random location: city (50%), home (15%), building+city (35%)."""
    r = rng.random()
    if r < 0.50:
        return rng.choice(templates["cities"])
    elif r < 0.65:
        return rng.choice(templates["home_locations"])
    else:
        building = rng.choice(templates["buildings"])
        city = rng.choice(templates["cities"])
        prep = rng.choice(_CITY_PREPS)
        if prep:
            return f"{building} {prep} {city}"
        return f"{building} {city}"


def _make_entry(
    loc_from: str, loc_to: str, purpose: str, prep_to: str, rng: random.Random
) -> list[tuple[str, str]]:
    """'van {from} naar {to} [voor] {purpose}'."""
    tokens: list[tuple[str, str]] = []
    tokens.extend(tagged_o(["van"]))
    tokens.extend(tagged_entity(loc_from.split(), B_LF, I_LF))
    tokens.extend(tagged_o([prep_to]))
    tokens.extend(tagged_entity(loc_to.split(), B_LT, I_LT))
    if purpose:
        if rng.random() < 0.5:
            tokens.extend(tagged_o(["voor"]))
        tokens.extend(tagged_entity(purpose.split(), B_P, I_P))
    return tokens


def _make_purpose_first(
    loc_from: str, loc_to: str, purpose: str, prep_to: str
) -> list[tuple[str, str]]:
    """'{purpose} van {from} naar {to}'."""
    tokens: list[tuple[str, str]] = []
    tokens.extend(tagged_entity(purpose.split(), B_P, I_P))
    tokens.extend(tagged_o(["van"]))
    tokens.extend(tagged_entity(loc_from.split(), B_LF, I_LF))
    tokens.extend(tagged_o([prep_to]))
    tokens.extend(tagged_entity(loc_to.split(), B_LT, I_LT))
    return tokens


def generate_examples(templates: dict, rng: random.Random, count: int) -> list[dict]:
    purposes = templates["purposes"]
    connectors = templates["connectors"]
    preps_to = templates["prepositions_to"]

    examples = []
    for _ in range(count):
        num_entries = rng.choices([1, 2, 3], weights=[40, 45, 15])[0]
        all_tokens: list[tuple[str, str]] = []

        for entry_idx in range(num_entries):
            loc_from = _pick_location(templates, rng)
            loc_to = _pick_location(templates, rng)
            # Avoid same from/to
            while loc_to == loc_from:
                loc_to = _pick_location(templates, rng)
            purpose = rng.choice(purposes) if rng.random() < 0.75 else ""
            prep_to = rng.choice(preps_to)

            if entry_idx > 0:
                conn = rng.choice(connectors)
                if conn:
                    all_tokens.extend(tagged_o([conn]))

            r = rng.random()
            if r < 0.15 and purpose:
                all_tokens.extend(_make_purpose_first(loc_from, loc_to, purpose, prep_to))
            else:
                all_tokens.extend(_make_entry(loc_from, loc_to, purpose, prep_to, rng))

        all_tokens = add_stt_noise(all_tokens, rng)
        if all_tokens:
            examples.append({
                "tokens": [t for t, _ in all_tokens],
                "ner_tags": [t for _, t in all_tokens],
            })

    return examples


def main():
    templates = load_templates("km_templates.json")
    write_splits("km", generate_examples, templates, random.Random(42))


if __name__ == "__main__":
    main()
