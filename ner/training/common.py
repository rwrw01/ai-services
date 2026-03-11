"""Shared utilities for synthetic NER training data generation."""

import json
import random
from pathlib import Path

TAG_O = "O"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"


def load_templates(filename: str) -> dict:
    with open(SCRIPT_DIR / "templates" / filename) as f:
        return json.load(f)


def tagged_entity(
    words: list[str], b_tag: str, i_tag: str
) -> list[tuple[str, str]]:
    """Tag a multi-word entity with BIO tags."""
    if not words:
        return []
    return [(words[0], b_tag)] + [(w, i_tag) for w in words[1:]]


def tagged_o(words: list[str]) -> list[tuple[str, str]]:
    """Tag filler words as O. Splits multi-word strings into individual tokens."""
    result = []
    for w in words:
        if not w:
            continue
        for part in w.split():
            result.append((part, TAG_O))
    return result


def add_stt_noise(
    tokens: list[tuple[str, str]], rng: random.Random
) -> list[tuple[str, str]]:
    """Simulate typical Canary STT output variations."""
    result = []
    for word, tag in tokens:
        if rng.random() < 0.10:
            word = word.capitalize()
        elif rng.random() < 0.05:
            word = word.lower()
        result.append((word, tag))

    # Occasionally drop or duplicate an O-tagged filler word
    if rng.random() < 0.03 and len(result) > 3:
        o_indices = [i for i, (_, t) in enumerate(result) if t == TAG_O]
        if o_indices:
            idx = rng.choice(o_indices)
            if rng.random() < 0.5:
                result.pop(idx)
            else:
                result.insert(idx, result[idx])

    return result


def write_splits(
    domain: str,
    generate_fn,
    templates: dict,
    rng: random.Random,
    train_count: int = 4000,
    val_count: int = 500,
    test_count: int = 500,
) -> None:
    """Generate train/val/test splits and write to JSONL files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for split, count in [("train", train_count), ("val", val_count), ("test", test_count)]:
        examples = generate_fn(templates, rng, count)
        path = DATA_DIR / f"{domain}_{split}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(examples)} examples to {path}")
