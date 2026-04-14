"""Curated 8-class 20newsgroups subset used as REAP's Tier 2 golden fixture.

Why this fixture
----------------
Synthetic Gaussian fixtures do not demonstrate REAP's advantage over
Procrustes because geometric blobs differ between seeds only by near-
rigid transforms, which Procrustes aligns well. The phenomenon REAP
targets is *semantic* overlap — regions of the embedding space where
multiple plausible clusterings exist — which shows up only on real
text with genuine topic ambiguity.

20newsgroups (Lang, 1995) is the standard topic-modeling benchmark.
We curate an 8-class subset with a pre-registered structure:

* 4 clearly separated topics: sci.space, rec.sport.hockey,
  comp.graphics, soc.religion.christian.
* Overlap pair A (politics, shared political vocabulary):
  talk.politics.guns, talk.politics.mideast.
* Overlap pair B (personal computing hardware, shared tech vocabulary):
  comp.sys.ibm.pc.hardware, comp.sys.mac.hardware.

50 documents per class are subsampled with a fixed RNG seed, yielding
400 documents — matching the sample size of the blob fixture. Each
document has sklearn's `remove=('headers', 'footers', 'quotes')` strip
applied, is truncated to 2000 characters, and is embedded with
`sentence-transformers/all-MiniLM-L6-v2` (384-d, L2-normalized).

Caching
-------
First load downloads the dataset (~14 MB) and embeddings are computed
(~30 s on CPU). Subsequent loads read from
`~/.cache/reap/datasets/20ng_golden_<hash>_minilm-l6-v2.npz`.

Determinism
-----------
Given the same sampling seed and the same MiniLM model version, the
embedding matrix is byte-identical. Loader verifies this via SHA256 on
every read.

Side effects: creates `~/.cache/reap/datasets/` on first call; may
trigger a ~90 MB Hugging Face model download for the embedding model
on first use.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from reap.datasets._schema import DatasetMetadata, DatasetSnapshot

logger = logging.getLogger(__name__)

GOLDEN_20NG_CLASSES: list[str] = [
    "sci.space",
    "rec.sport.hockey",
    "comp.graphics",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
]
OVERLAP_PAIRS: list[tuple[int, int]] = [(4, 5), (6, 7)]
N_DOCS_PER_CLASS: int = 50
MIN_DOC_CHARS: int = 100
MAX_DOC_CHARS: int = 2000
EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLING_SEED: int = 20260413


def _get_cache_dir() -> Path:
    """Return the REAP dataset cache dir, creating it if needed."""
    cache = Path.home() / ".cache" / "reap" / "datasets"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _fetch_and_subsample() -> tuple[list[str], np.ndarray]:
    """Fetch, header-strip, class-filter, and deterministically subsample.

    Returns
    -------
    texts : list of 400 cleaned document strings in canonical class order.
    labels : (400,) int64 array with our canonical class IDs (0..7).
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = cast(
        Any,
        fetch_20newsgroups(
            subset="all",
            categories=GOLDEN_20NG_CLASSES,
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        ),
    )
    raw_texts: list[str] = list(bunch.data)
    raw_labels = np.asarray(bunch.target, dtype=np.int64)

    name_to_our_id = {name: i for i, name in enumerate(GOLDEN_20NG_CLASSES)}
    our_labels = np.array(
        [name_to_our_id[bunch.target_names[t]] for t in raw_labels],
        dtype=np.int64,
    )

    rng = np.random.default_rng(SAMPLING_SEED)
    selected: list[int] = []
    for class_id in range(len(GOLDEN_20NG_CLASSES)):
        class_mask = our_labels == class_id
        class_indices = np.where(class_mask)[0]
        usable = [
            int(i)
            for i in class_indices
            if len(raw_texts[int(i)].strip()) >= MIN_DOC_CHARS
        ]
        if len(usable) < N_DOCS_PER_CLASS:
            raise ValueError(
                f"Class {GOLDEN_20NG_CLASSES[class_id]!r} has only "
                f"{len(usable)} usable docs after filtering (need "
                f"{N_DOCS_PER_CLASS}). Lower MIN_DOC_CHARS or pick a "
                "different subset."
            )
        chosen = rng.choice(
            np.asarray(usable, dtype=np.int64),
            size=N_DOCS_PER_CLASS,
            replace=False,
        )
        selected.extend(int(i) for i in chosen)

    selected.sort()
    texts = [raw_texts[i].strip()[:MAX_DOC_CHARS] for i in selected]
    labels = our_labels[selected]
    return texts, labels


def _corpus_sha(texts: list[str]) -> str:
    """SHA256 of the concatenated corpus — stable across platforms."""
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")  # ASCII record separator as unambiguous delimiter
    return h.hexdigest()


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed with sentence-transformers MiniLM, L2-normalized output."""
    try:
        from sentence_transformers import (  # pyright: ignore[reportMissingImports]
            SentenceTransformer,
        )
    except ImportError as err:  # pragma: no cover — covered by skipif in tests
        raise ImportError(
            "reap.datasets.load_golden_text requires sentence-transformers. "
            "Install with: pip install 'reap-topics[text-fixtures]'"
        ) from err

    model = SentenceTransformer(EMBEDDING_MODEL_ID)
    embs = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embs, dtype=np.float32)


def load_golden_text() -> DatasetSnapshot:
    """Load the REAP Tier 2 golden text fixture (8-class 20newsgroups).

    Returns a `DatasetSnapshot` with 400 documents, 384-d embeddings,
    integer labels in {0..7}, and full provenance metadata including
    the SHA256 of the embedding payload.

    Side effects
    ------------
    * Creates `~/.cache/reap/datasets/` on first call.
    * First call downloads the 20newsgroups archive (~14 MB) via sklearn.
    * First call downloads the MiniLM model (~90 MB) via huggingface.
    * First call computes embeddings (~30 s on CPU); result is cached.
    """
    texts, labels = _fetch_and_subsample()
    corpus_hash = _corpus_sha(texts)[:16]
    cache_path = _get_cache_dir() / f"20ng_golden_{corpus_hash}_minilm-l6-v2.npz"

    if cache_path.exists():
        logger.info("Loading cached 20ng embeddings from %s", cache_path)
        cached = np.load(cache_path)
        embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
    else:
        logger.info(
            "Embedding %d documents with %s (first run, ~30s on CPU)",
            len(texts),
            EMBEDDING_MODEL_ID,
        )
        embs_f32 = _embed_texts(texts)
        np.savez(cache_path, embeddings=embs_f32)
        embeddings = embs_f32.astype(np.float64)

    payload_sha = hashlib.sha256(
        np.ascontiguousarray(embeddings).tobytes() + labels.tobytes()
    ).hexdigest()

    meta = DatasetMetadata(
        name="golden_text_20ng_8class",
        version="1.0.0",
        n_samples=len(texts),
        embedding_dim=int(embeddings.shape[1]),
        embedding_model=f"{EMBEDDING_MODEL_ID} (L2-normalized)",
        preprocessing_version=(
            "1.0.0-remove-headers-footers-quotes-truncate-2000"
        ),
        sha256=payload_sha,
        license="public-domain (20newsgroups, Lang 1995)",
        citation=(
            "@inproceedings{lang1995newsweeder,"
            " title={NewsWeeder: Learning to Filter Netnews},"
            " author={Lang, Ken},"
            " booktitle={Proc. 12th Int. Conf. on Machine Learning},"
            " year={1995},"
            " pages={331-339}"
            "}"
        ),
        source_url=(
            "https://scikit-learn.org/stable/datasets/real_world.html"
            "#the-20-newsgroups-text-dataset"
        ),
        notes=(
            "Curated 8-class subset. Classes 0-3 are clearly separated "
            "(sci.space, rec.sport.hockey, comp.graphics, "
            "soc.religion.christian). Classes 4-5 form overlap pair A "
            "(talk.politics.guns vs talk.politics.mideast). Classes 6-7 "
            "form overlap pair B (comp.sys.ibm.pc.hardware vs "
            f"comp.sys.mac.hardware). {N_DOCS_PER_CLASS} docs per class, "
            f"sampled with seed={SAMPLING_SEED}. Corpus SHA256 "
            f"prefix: {corpus_hash}."
        ),
    )

    return DatasetSnapshot(
        embeddings=embeddings, texts=texts, labels=labels, metadata=meta
    )
