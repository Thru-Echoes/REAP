"""20-Newsgroups out-of-sample loader (12 held-out classes × 50 docs = 600).

The OOS half of the class-held-out manuscript validation pair. The
reference snapshot (`load_twenty_newsgroups_reference()`) covers the
same 8 classes as the golden fixture; this snapshot covers the
**remaining 12 classes** of the 20-class corpus, providing a true
class-held-out test of REAP's projection head — every OOS document
belongs to a topic the consensus reducer has never seen.

Held-out 12-class subset (pre-registered, disjoint from reference):

* ``alt.atheism``
* ``comp.os.ms-windows.misc``
* ``comp.windows.x``
* ``misc.forsale``
* ``rec.autos``
* ``rec.motorcycles``
* ``rec.sport.baseball``
* ``sci.crypt``
* ``sci.electronics``
* ``sci.med``
* ``talk.politics.misc``
* ``talk.religion.misc``

50 documents per class are subsampled with the same RNG seed used by
the reference loader, yielding 600 documents. Each document has
sklearn's ``remove=('headers', 'footers', 'quotes')`` strip applied,
is truncated to 2000 characters, and is embedded with
``sentence-transformers/all-MiniLM-L6-v2`` (384-d, L2-normalized) —
identical preprocessing and embedding pipeline to the reference
snapshot, so the only systematic difference between the two is the
class membership.

Label offset
------------
To make the reference and OOS label spaces unambiguously disjoint (a
property exercised by `tests/test_twenty_newsgroups_validation.py`),
OOS labels start at `LABEL_OFFSET = 100` rather than 0: class
``alt.atheism`` → 100, ``comp.os.ms-windows.misc`` → 101, …,
``talk.religion.misc`` → 111. Downstream consumers can recover the
class name from
``OOS_20NG_CLASSES[label - LABEL_OFFSET]``.

Caching
-------
First load downloads the 20newsgroups archive (~14 MB; shared with
the reference loader's cache) and the MiniLM model (~90 MB), then
computes embeddings (~45 s on CPU for 600 documents). Subsequent loads
read from
``~/.cache/reap/datasets/20ng_oos_<hash>_minilm-l6-v2.npz``.

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

OOS_20NG_CLASSES: list[str] = [
    "alt.atheism",
    "comp.os.ms-windows.misc",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "talk.politics.misc",
    "talk.religion.misc",
]
LABEL_OFFSET: int = 100
N_DOCS_PER_CLASS: int = 50
MIN_DOC_CHARS: int = 100
MAX_DOC_CHARS: int = 2000
EMBEDDING_MODEL_ID: str = "sentence-transformers/all-MiniLM-L6-v2"
SAMPLING_SEED: int = 20260413
DATASET_NAME: str = "twenty_newsgroups_oos"
DATASET_VERSION: str = "1.0.0"
EXPECTED_N_SAMPLES: int = N_DOCS_PER_CLASS * len(OOS_20NG_CLASSES)  # 600
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

    Pulls the 12 held-out classes from the full 20newsgroups corpus via
    sklearn (``remove=('headers', 'footers', 'quotes')``), filters out
    documents shorter than `MIN_DOC_CHARS` after stripping, samples
    `N_DOCS_PER_CLASS` documents per class using `SAMPLING_SEED`, and
    truncates each survivor to `MAX_DOC_CHARS`. Returned labels are
    offset by `LABEL_OFFSET` to avoid collision with the reference
    snapshot's 0..7 label space.

    Returns
    -------
    texts : list of 600 cleaned document strings in canonical class order.
    labels : (600,) int64 array with class IDs in ``{100..111}``.

    Side effects: downloads the 20newsgroups archive on first call.
    """
    from sklearn.datasets import fetch_20newsgroups

    bunch = cast(
        Any,
        fetch_20newsgroups(
            subset="all",
            categories=OOS_20NG_CLASSES,
            remove=("headers", "footers", "quotes"),
            shuffle=False,
        ),
    )
    raw_texts: list[str] = list(bunch.data)
    raw_labels = np.asarray(bunch.target, dtype=np.int64)

    name_to_our_id = {name: i for i, name in enumerate(OOS_20NG_CLASSES)}
    our_labels = np.array(
        [name_to_our_id[bunch.target_names[t]] for t in raw_labels],
        dtype=np.int64,
    )

    rng = np.random.default_rng(SAMPLING_SEED)
    selected: list[int] = []
    for class_id in range(len(OOS_20NG_CLASSES)):
        class_mask = our_labels == class_id
        class_indices = np.where(class_mask)[0]
        usable = [
            int(i)
            for i in class_indices
            if len(raw_texts[int(i)].strip()) >= MIN_DOC_CHARS
        ]
        if len(usable) < N_DOCS_PER_CLASS:
            raise ValueError(
                f"Class {OOS_20NG_CLASSES[class_id]!r} has only "
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
    labels = (our_labels[selected] + LABEL_OFFSET).astype(np.int64)
    return texts, labels


def _corpus_sha(texts: list[str]) -> str:
    """SHA256 of the concatenated corpus — stable across platforms.

    Uses the ASCII record separator ``0x1e`` as an unambiguous delimiter
    between documents. Side effects: none.
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
            "reap.datasets.load_twenty_newsgroups_oos requires "
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


def load_twenty_newsgroups_oos() -> DatasetSnapshot:
    """Load the manuscript 20-Newsgroups OOS snapshot (12 held-out × 50).

    Returns a `DatasetSnapshot` with 600 documents, 384-d MiniLM
    embeddings, integer expert labels in ``{100..111}`` (offset to be
    disjoint from the reference snapshot's ``{0..7}`` label space), and
    full provenance metadata including the SHA256 of the embedding
    payload.

    The 12 classes are the 20-Newsgroups categories NOT used by
    `load_twenty_newsgroups_reference()`. Every document belongs to a
    topic the reference snapshot has not seen, making this corpus a
    true class-held-out OOS evaluation for REAP's projection head.

    Returns
    -------
    DatasetSnapshot
        ``embeddings.shape == (600, 384)``,
        ``labels.shape == (600,)`` with values in ``{100..111}``,
        ``texts`` is a list of 600 cleaned source strings,
        ``metadata`` is a fully-populated `DatasetMetadata`.

    Side effects
    ------------
    * Creates ``~/.cache/reap/datasets/`` on first call.
    * First call downloads the 20newsgroups archive (~14 MB) via sklearn
      (shared with the reference loader's sklearn cache).
    * First call downloads the MiniLM model (~90 MB) via huggingface.
    * First call computes embeddings (~45 s on CPU); result is cached.
    """
    texts, labels = _fetch_and_subsample()
    corpus_hash = _corpus_sha(texts)[:16]
    cache_path = _get_cache_dir() / f"20ng_oos_{corpus_hash}_minilm-l6-v2.npz"

    if cache_path.exists():
        logger.info("Loading cached 20ng OOS embeddings from %s", cache_path)
        cached = np.load(cache_path)
        embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
    else:
        logger.info(
            "Embedding %d documents with %s (first run, ~45s on CPU)",
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
            "Manuscript OOS snapshot: 12 held-out 20newsgroups classes "
            f"({N_DOCS_PER_CLASS} docs per class, "
            f"sampled with seed={SAMPLING_SEED}). "
            "Disjoint from load_twenty_newsgroups_reference(). Labels are "
            f"offset by LABEL_OFFSET={LABEL_OFFSET} so reference labels "
            "{0..7} and OOS labels {100..111} cannot collide. Held-out "
            "classes: " + ", ".join(OOS_20NG_CLASSES) + ". "
            f"Corpus SHA256 prefix: {corpus_hash}."
        ),
    )

    return DatasetSnapshot(
        embeddings=embeddings, texts=texts, labels=labels, metadata=meta
    )
