import asyncio
import logging

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from app.config import DOMAIN_LABELS, MODELS_DIR
from app.engines.base import NEREngine
from app.postprocess import bio_to_km_entries, bio_to_uren_entries

logger = logging.getLogger(__name__)

_DOMAIN_POSTPROCESSORS = {
    "uren": bio_to_uren_entries,
    "km": bio_to_km_entries,
}

_inference_sem = asyncio.Semaphore(4)


class RobBERTEngine(NEREngine):
    name = "robbertje"

    def __init__(self):
        self._models: dict[str, AutoModelForTokenClassification] = {}
        self._tokenizers: dict[str, AutoTokenizer] = {}
        self._id2label: dict[str, dict[int, str]] = {}
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        for domain, labels in DOMAIN_LABELS.items():
            model_path = MODELS_DIR / domain
            if not model_path.exists():
                logger.warning("No model for domain '%s' at %s", domain, model_path)
                continue
            logger.info("Loading NER model for '%s' from %s", domain, model_path)
            tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            model = AutoModelForTokenClassification.from_pretrained(
                str(model_path), use_safetensors=True,
            )
            model.to(self._device)
            model.eval()
            self._tokenizers[domain] = tokenizer
            self._models[domain] = model
            self._id2label[domain] = dict(enumerate(labels))
            logger.info("Loaded '%s' (%d labels)", domain, len(labels))

    async def extract(self, text: str, domain: str) -> dict:
        if domain not in self._models:
            raise ValueError(f"Domain '{domain}' not loaded")

        async with _inference_sem:
            predictions = await asyncio.to_thread(self._predict, text, domain)
        postprocess = _DOMAIN_POSTPROCESSORS.get(domain)
        entries = postprocess(predictions) if postprocess else []

        return {"entries": entries, "engine": self.name, "domain": domain}

    async def health(self) -> dict:
        return {
            "engine": self.name,
            "device": self._device,
            "domains": list(self._models.keys()),
        }

    def _predict(self, text: str, domain: str) -> list[tuple[str, str]]:
        """Run NER prediction, returns list of (token, label) pairs."""
        tokenizer = self._tokenizers[domain]
        model = self._models[domain]
        id2label = self._id2label[domain]

        words = text.strip().split()
        encoding = tokenizer(
            words, is_split_into_words=True,
            return_tensors="pt", truncation=True, padding=True,
        )
        word_ids = encoding.word_ids(batch_index=0)
        inputs = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)[0].tolist()

        result: list[tuple[str, str]] = []
        prev_word_id = None
        for idx, word_id in enumerate(word_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            result.append((words[word_id], id2label.get(pred_ids[idx], "O")))
            prev_word_id = word_id

        return result
