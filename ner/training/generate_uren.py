"""Generate synthetic NER training data for the 'uren' domain.

Usage:
    python -m training.generate_uren
"""

import random

from training.common import (
    add_stt_noise,
    load_templates,
    tagged_entity,
    tagged_o,
    write_splits,
)

B_TS, I_TS = "B-TIME_START", "I-TIME_START"
B_TE, I_TE = "B-TIME_END", "I-TIME_END"
B_D, I_D = "B-DESC", "I-DESC"


def _make_entry(
    start_expr: str, end_expr: str, desc: str, prep: str = ""
) -> list[tuple[str, str]]:
    """'van {start} tot {end} [prep] {desc}'."""
    tokens: list[tuple[str, str]] = []
    tokens.extend(tagged_o(["van"]))
    tokens.extend(tagged_entity(start_expr.split(), B_TS, I_TS))
    tokens.extend(tagged_o(["tot"]))
    tokens.extend(tagged_entity(end_expr.split(), B_TE, I_TE))
    if prep:
        tokens.extend(tagged_o([prep]))
    tokens.extend(tagged_entity(desc.split(), B_D, I_D))
    return tokens


def _make_desc_first(
    start_expr: str, end_expr: str, desc: str
) -> list[tuple[str, str]]:
    """'{desc} van {start} tot {end}'."""
    tokens: list[tuple[str, str]] = []
    tokens.extend(tagged_entity(desc.split(), B_D, I_D))
    tokens.extend(tagged_o(["van"]))
    tokens.extend(tagged_entity(start_expr.split(), B_TS, I_TS))
    tokens.extend(tagged_o(["tot"]))
    tokens.extend(tagged_entity(end_expr.split(), B_TE, I_TE))
    return tokens


def generate_examples(templates: dict, rng: random.Random, count: int) -> list[dict]:
    time_map = templates["time_expressions"]
    descriptions = templates["descriptions"]
    connectors = templates["connectors"]
    prepositions = templates["prepositions_desc"]
    all_times = list(time_map.keys())

    examples = []
    for _ in range(count):
        num_entries = rng.choices([1, 2, 3, 4], weights=[30, 40, 25, 5])[0]
        time_indices = sorted(rng.sample(range(len(all_times)), min(num_entries * 2, len(all_times))))
        time_keys = [all_times[i] for i in time_indices]

        all_tokens: list[tuple[str, str]] = []
        for entry_idx in range(num_entries):
            si, ei = entry_idx * 2, entry_idx * 2 + 1
            if ei >= len(time_keys):
                break

            start_expr = rng.choice(time_map[time_keys[si]])
            end_expr = rng.choice(time_map[time_keys[ei]])
            desc = rng.choice(descriptions)

            if entry_idx > 0:
                conn = rng.choice(connectors)
                if conn:
                    all_tokens.extend(tagged_o([conn]))

            if rng.random() < 0.15:
                all_tokens.extend(_make_desc_first(start_expr, end_expr, desc))
            else:
                all_tokens.extend(_make_entry(start_expr, end_expr, desc, rng.choice(prepositions)))

        all_tokens = add_stt_noise(all_tokens, rng)
        if all_tokens:
            examples.append({
                "tokens": [t for t, _ in all_tokens],
                "ner_tags": [t for _, t in all_tokens],
            })

    return examples


def main():
    templates = load_templates("uren_templates.json")
    write_splits("uren", generate_examples, templates, random.Random(42))


if __name__ == "__main__":
    main()
