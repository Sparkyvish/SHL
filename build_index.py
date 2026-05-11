"""
build_index.py  –  Embeds catalog.json and writes a FAISS index + metadata.

Run after scraper.py:
    python build_index.py

Writes:
    index.faiss   – FAISS flat L2 index
    index_meta.json – list of assessment dicts aligned with FAISS row IDs
"""

import json
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"  # 80 MB, fast, good quality for retrieval
CATALOG_PATH = "catalog.json"
INDEX_PATH = "index.faiss"
META_PATH = "index_meta.json"


def _build_doc_text(a: dict) -> str:
    """
    Build a rich text representation of an assessment to embed.
    Include all fields that a hiring manager might search for.
    """
    parts = [
        f"Name: {a['name']}",
        f"Description: {a['description']}",
        f"Test types: {', '.join(a.get('test_type_labels', []))}",
        f"Type codes: {', '.join(a.get('test_types', []))}",
    ]
    if a.get("remote_testing"):
        parts.append("Supports remote testing: yes")
    if a.get("adaptive_irt"):
        parts.append("Adaptive / IRT: yes")
    if a.get("duration_minutes"):
        parts.append(f"Duration: {a['duration_minutes']} minutes")
    if a.get("job_levels"):
        parts.append(f"Job levels: {', '.join(a['job_levels'])}")
    if a.get("languages"):
        parts.append(f"Languages: {', '.join(a['languages'][:5])}")
    return " | ".join(parts)


def build_index(
    catalog_path: str = CATALOG_PATH,
    index_path: str = INDEX_PATH,
    meta_path: str = META_PATH,
) -> None:
    log.info("Loading catalog from %s …", catalog_path)
    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    log.info("Loaded %d assessments.", len(catalog))

    log.info("Loading embedding model: %s …", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    docs = [_build_doc_text(a) for a in catalog]
    log.info("Embedding %d documents …", len(docs))
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    # L2-normalise → cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalised vecs = cosine sim
    index.add(embeddings)
    log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

    faiss.write_index(index, index_path)
    log.info("Saved FAISS index → %s", index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    log.info("Saved metadata → %s", meta_path)


if __name__ == "__main__":
    build_index()
