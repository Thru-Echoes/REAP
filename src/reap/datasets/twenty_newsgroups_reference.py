"""20-Newsgroups reference loader (8 classes × 100 docs = 800 documents).

Manuscript validation dataset — Tier-3, third dataset in the cross-corpus
replication design (alongside AI-art and Korean forest). Provides an
**in-distribution reference corpus** that is paired with
`load_twenty_newsgroups_oos()`, a held-out 12-class corpus used to
exercise REAP's out-of-sample projection head on truly unseen topical
content.

Why this fixture
----------------
The existing `load_golden_text()` fixture (8 classes × 50 docs = 400)
is used for the Tier-1/2/3 *golden-validation* tests and is sized to
match the synthetic blob fixture. For the manuscript's third real-
dataset comparison we need a larger snapshot, comparable in size to
Korean forest (905) and AI-art (1,736). We therefore use the same
curated 8-class subset as the golden fixture but draw 100 docs/class
= 800 documents, with the remaining 12 classes reserved for the OOS
snapshot. Same sampling seed, same preprocessing, same MiniLM model.

Curated 8-class subset (pre-registered, identical to `load_golden_text`):

* 4 clearly separated topics: ``sci.space``, ``rec.sport.hockey``,
  ``comp.graphics``, ``soc.religion.christian``.
* Overlap pair A (politics, shared political vocabulary):
  ``talk.politics.guns``, ``talk.politics.mideast``.
* Overlap pair B (personal computing hardware, shared tech vocabulary):
  ``comp.sys.ibm.pc.hardware``, ``comp.sys.mac.hardware``.

Each document has sklearn's ``remove=('headers', 'footers', 'quotes')``
strip applied, is truncated to 2000 characters, and is embedded with
``sentence-transformers/all-MiniLM-L6-v2`` (384-d, L2-normalized).

Caching
-------
First load downloads the 20newsgroups archive (~14 MB) via sklearn and
the MiniLM model (~90 MB) via Hugging Face, then computes embeddings
(~60 s on CPU for 800 documents). Subsequent loads read from
``~/.cache/reap/datasets/20ng_reference_<hash>_minilm-l6-v2.npz``.

Determinism
-----------
Given the same `SAMPLING_SEED` and the same MiniLM model version, the
embedding matrix is byte-identical. The SHA256 of the embedding payload
is recorded in `DatasetMetadata.sha256` for downstream integrity checks.

Side effects: creates ``~/.cache/reap/datasets/`` on first call; may
trigger a ~90 MB Hugging Face model download on first use.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

from reap.datasets._schema import DatasetMetadata, DatasetSnapshot

logger = logging.getLogger(__name__)

REFERENCE_20NG_CLASSES: list[str] = [
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
N_DOCS_PER_CLASS: int = 100
MIN_DOC_CHARS: int = 100
MAX_DOC_CHARS: int = 2000
EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLING_SEED: int = 20260413
DATASET_NAME: str = "twenty_newsgroups_reference"
DATASET_VERSION: str = "1.0.0"
EXPECTED_N_SAMPLES: int = N_DOCS_PER_CLASS * len(REFERENCE_20NG_CLASSES)  # 800
EXPECTED_EMBEDDING_DIM: int = 384


def _get_cache_dir() -> Path:
    """Return the REAP dataset cache dir, creating it if needed.

    Side effects: creates ``~/.cache/reap/datasets`` if absent.
    """
    cache = Path.home() / ".cache" / "reap" / "datasets"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _fetch_and_subsample() -> tuple[list[str], np.ndarray]:
    """Fetch, header-strip, class-filter, and deterministically subsample.

    Pulls the 8 reference classes from the full 20newsgroups corpus via
    sklearn (``remove=('headers', 'footers', 'quotes')``), filters out
    documents shorter than `MIN_DOC_CHARS` after stripping, samples
    `N_DOCS_PER_CLASS` documents per class using `SAMPLING_SEED`, and
    truncates each survivor to `MAX_DOC_CHARS`.

    Returns
    -------
    texts : list of 800 cleaned document strings in canonical class order.
    labels : (800,) int64 array with our canonical class IDs (0..7).

    Side effects: downloads the 20newsgroups archive on first call.
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = cast(
        Any,
        fetch_20newsgroups(
            subset="all",
            categories=REFERENCE_20NG_CLASSES,
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        ),
    )
    raw_texts: list[str] = list(bunch.data)
    raw_labels = np.asarray(bunch.target, dtype=np.int64)

    name_to_our_id = {name: i for i, name in enumerate(REFERENCE_20NG_CLASSES)}
    our_labels = np.array(
        [name_to_our_id[bunch.target_names[t]] for t in raw_labels],
        dtype=np.int64,
    )

    rng = np.random.default_rng(SAMPLING_SEED)
    selected: list[int] = []
    for class_id in range(len(REFERENCE_20NG_CLASSES)):
        class_mask = our_labels == class_id
        class_indices = np.where(class_mask)[0]
        usable = [
            int(i)
            for i in class_indices
            if len(raw_texts[int(i)].strip()) >= MIN_DOC_CHARS
        ]
        if len(usable) < N_DOCS_PER_CLASS:
            raise ValueError(
                f"Class {REFERENCE_20NG_CLASSES[class_id]!r} has only "
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
    """SHA256 of the concatenated corpus — stable across platforms.

    Uses the ASCII record separator ``0x1e`` as an unambiguous delimiter
    between documents so that ``["ab", "cd"]`` and ``["a", "bcd"]`` hash
    differently. Side effects: none.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Embed with sentence-transformers MiniLM, L2-normalized output.

    Returns
    -------
    (n_samples, 384) float32 array; each row is unit-norm.

    Raises
    ------
    ImportError
        ``sentence-transformers`` not installed.

    Side effects: triggers a ~90 MB model download on first use.
    """
    try:
        from sentence_transformers import (  # pyright: ignore[reportMissingImports]
            SentenceTransformer,
        )
    except ImportError as err:  # pragma: no cover — covered by skipif in tests
        raise ImportError(
            "reap.datasets.load_twenty_newsgroups_reference requires "
            "sentence-transformers. Install with: "
            "pip install 'reap-topics[text-fixtures]'"
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


def load_twenty_newsgroups_reference() -> DatasetSnapshot:
    """Load the manuscript 20-Newsgroups reference snapshot (8 cls × 100 docs).

    Returns a `DatasetSnapshot` with 800 documents, 384-d MiniLM
    embeddings, integer expert labels in ``{0..7}``, and full provenance
    metadata including the SHA256 of the embedding payload.

    The 8 classes used are the same as `load_golden_text()` — the
    overlap-pair design (sci.space, rec.sport.hockey, comp.graphics,
    soc.religion.christian, talk.politics.guns, talk.politics.mideast,
    comp.sys.ibm.pc.hardware, comp.sys.mac.hardware). The remaining
    12 classes are reserved for `load_twenty_newsgroups_oos()`.

    Returns
    -------
    DatasetSnapshot
        ``embeddings.shape == (800, 384)``,
        ``labels.shape == (800,)`` with values in ``{0..7}``,
        ``texts`` is a list of 800 cleaned source strings,
        ``metadata`` is a fully-populated `DatasetMetadata`.

    Side effects
    ------------
    * Creates ``~/.cache/reap/datasets/`` on first call.
    * First call downloads the 20newsgroups archive (~14 MB) via sklearn.
    * First call downloads the MiniLM model (~90 MB) via huggingface.
    * First call computes embeddings (~60 s on CPU); result is cached.
    """
    texts, labels = _fetch_and_subsample()
    corpus_hash = _corpus_sha(texts)[:16]
    cache_path = _get_cache_dir() / f"20ng_reference_{corpus_hash}_minilm-l6-v2.npz"

    if cache_path.exists():
        logger.info("Loading cached 20ng reference embeddings from %s", cache_path)
        cached = np.load(cache_path)
        embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
    else:
        logger.info(
            "Embedding %d documents with %s (first run, ~60s on CPU)",
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
        name=DATASET_NAME,
        version=DATASET_VERSION,
        n_samples=len(texts),
        embedding_dim=int(embeddings.shape[1]),
        embedding_model=f"{EMBEDDING_MODEL_ID} (L2-normalized)",
        preprocessing_version=(
            "1.0.0-remove-headers-footers-quotes-truncate-2000"
        ),
        sha256=payload_sha,
        license="by-permission",
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
            "Manuscript reference snapshot: 8-class 20newsgroups subset, "
            f"{N_DOCS_PER_CLASS} docs per class, "
            f"sampled with seed={SAMPLING_SEED}. Classes 0-3 are "
            "clearly separated (sci.space, rec.sport.hockey, comp.graphics, "
            "soc.religion.christian). Classes 4-5 form overlap pair A "
            "(talk.politics.guns vs talk.politics.mideast). Classes 6-7 "
            "form overlap pair B (comp.sys.ibm.pc.hardware vs "
            "comp.sys.mac.hardware). Paired with "
            "load_twenty_newsgroups_oos() for class-held-out OOS projection "
            f"validation. Corpus SHA256 prefix: {corpus_hash}."
        ),
    )

    return DatasetSnapshot(
        embeddings=embeddings, texts=texts, labels=labels, metadata=meta
    )
