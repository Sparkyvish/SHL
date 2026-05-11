"""
retriever.py  –  Thin wrapper around the FAISS index for semantic retrieval.

Loaded once at process start; subsequent calls are in-memory lookups.
"""

import json
import logging
from functools import lru_cache
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "index.faiss"
META_PATH = "index_meta.json"
TOP_K = 15  # retrieve more than we'll show; agent prunes to ≤10


@lru_cache(maxsize=1)
def _load_resources():
    """Load model, index, and metadata once and cache."""
    log.info("Loading embedding model …")
    model = SentenceTransformer(MODEL_NAME)

    log.info("Loading FAISS index from %s …", INDEX_PATH)
    index = faiss.read_index(INDEX_PATH)

    log.info("Loading metadata from %s …", META_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)

    log.info("Retriever ready: %d assessments indexed.", len(meta))
    return model, index, meta


def retrieve(query: str, k: int = TOP_K) -> List[dict]:
    """
    Return the top-k most semantically similar assessments for `query`.
    Each dict is the full assessment record from the catalog.
    """
    model, index, meta = _load_resources()

    vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(vec)

    scores, ids = index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        item = dict(meta[idx])
        item["_score"] = float(score)
        results.append(item)

    return results


def get_by_name(name: str) -> List[dict]:
    """Fuzzy name lookup – useful for comparison queries like 'what is OPQ32?'."""
    _, _, meta = _load_resources()
    name_lower = name.lower()
    return [a for a in meta if name_lower in a["name"].lower()]


def get_all() -> List[dict]:
    """Return the full catalog (for system-prompt context on small catalogs)."""
    _, _, meta = _load_resources()
    return list(meta)
