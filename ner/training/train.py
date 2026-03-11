"""Fine-tune RobBERTje for token classification (NER) on a domain.

Usage:
    python -m training.train --domain uren
    python -m training.train --domain km
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BASE_MODEL, DOMAIN_LABELS, MODELS_DIR


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=list(DOMAIN_LABELS.keys()))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    # Lazy imports (heavy dependencies)
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    labels = DOMAIN_LABELS[args.domain]
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for i, l in enumerate(labels)}

    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_data = load_jsonl(data_dir / f"{args.domain}_train.jsonl")
    val_data = load_jsonl(data_dir / f"{args.domain}_val.jsonl")

    if not train_data:
        print(f"No training data found at {data_dir / f'{args.domain}_train.jsonl'}")
        sys.exit(1)

    print(f"Domain: {args.domain}")
    print(f"Labels: {labels}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    print(f"Base model: {BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    def tokenize_and_align(examples: dict) -> dict:
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding=False,
            max_length=args.max_length,
        )

        all_labels = []
        for i, ner_tags in enumerate(examples["ner_tags"]):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            prev_word_id = None
            for word_id in word_ids:
                if word_id is None:
                    label_ids.append(-100)
                elif word_id != prev_word_id:
                    tag = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                    label_ids.append(label2id.get(tag, 0))
                else:
                    # Subword continuation: use I- version or -100
                    tag = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                    if tag.startswith("B-"):
                        tag = "I-" + tag[2:]
                    label_ids.append(label2id.get(tag, -100))
                prev_word_id = word_id
            all_labels.append(label_ids)

        tokenized["labels"] = all_labels
        return tokenized

    train_ds = Dataset.from_list(train_data).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
    )
    val_ds = Dataset.from_list(val_data).map(
        tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"]
    )

    data_collator = DataCollatorForTokenClassification(tokenizer, padding=True)

    def compute_metrics(eval_pred):
        predictions, labels_arr = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        preds = np.argmax(predictions, axis=-1)

        # Flatten and filter -100
        true_labels = []
        pred_labels = []
        for pred_seq, label_seq in zip(preds, labels_arr):
            for p, l in zip(pred_seq, label_seq):
                if l != -100:
                    true_labels.append(l)
                    pred_labels.append(p)

        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0

        # Per-entity F1 (exclude O)
        entity_labels = [i for i, l in enumerate(labels) if l != "O"]
        tp = fp = fn = 0
        for eid in entity_labels:
            tp += sum(1 for t, p in zip(true_labels, pred_labels) if t == eid and p == eid)
            fp += sum(1 for t, p in zip(true_labels, pred_labels) if t != eid and p == eid)
            fn += sum(1 for t, p in zip(true_labels, pred_labels) if t == eid and p != eid)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    output_dir = MODELS_DIR / args.domain
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Final evaluation
    metrics = trainer.evaluate()
    print(f"\nFinal metrics: {metrics}")


if __name__ == "__main__":
    main()
