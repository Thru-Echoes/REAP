"""Cross-repo parity: REAP cache vs the source sibling repos.

REAP's dataset cache (``~/.cache/reap/datasets/``) is built from two
external research projects via ``scripts/build_datasets.py``:

* AI-art: ``~/Documents/Berkeley/Research/When-Algorithms-Meet-Artists``
* Korean forest: ``~/Documents/Berkeley/Research/green-narrative/hye_in``

This module is the explicit, audited check that **the cached arrays we
analyze match the source files in those sibling repos byte-for-byte**
(within the documented float32 cast for the AI-art and KF reference
embeddings, and float64-exact for the KF OOS pre-projected coords).
Every cross-repo loader's promise — that the snapshot is faithful to
the upstream source — is tested here.

Skip behavior
-------------
If a sibling repo is not present on disk, the affected test is
**skipped** with a clear ``pytest.skip`` reason, so this file can run
in CI environments that do not check out the sibling repos. In the
researcher's development environment (both sibling repos present), all
tests should pass.

The tests cover, for each cached dataset:

1. ``embeddings.npy`` ↔ source NPY/CSV/XLSX (shape, dtype, exact equality).
2. ``texts.json`` ↔ source text column (length, content).
3. ``labels.npy`` ↔ source label column (when present), under the same
   integer-encoding the loader applies.

Tests are intentionally rigid: any drift between the cache and the
sibling repo is a provenance failure and MUST not be silently silenced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

WAMA_ROOT = Path.home() / "Documents/Berkeley/Research/When-Algorithms-Meet-Artists"
HYEIN_ROOT = Path.home() / "Documents/Berkeley/Research/green-narrative/hye_in"
CACHE_ROOT = Path.home() / ".cache/reap/datasets"


def _require_sibling(root: Path) -> None:
    """Skip the test if the sibling repo isn't checked out at the expected path."""
    if not root.is_dir():
        pytest.skip(f"sibling repo not present at {root}")


# --------------------------------------------------------------------------- #
# AI-art: reference (1,736 chunks × 1,024 dims)                              #
# --------------------------------------------------------------------------- #


def test_ai_art_reference_embeddings_match_wama_source():
    """REAP's `ai_art` cache embeddings are byte-identical to the WAMA NPY."""
    _require_sibling(WAMA_ROOT)
    src = np.load(
        WAMA_ROOT / "figures/prefix_grid_search/prefix_embeddings_public.npy",
        allow_pickle=False,
    )
    cached = np.load(CACHE_ROOT / "ai_art/2026-04-14/embeddings.npy", allow_pickle=False)
    assert src.shape == cached.shape == (1736, 1024)
    assert src.dtype == cached.dtype == np.float32
    assert np.array_equal(src, cached), (
        f"max abs diff = {float(np.abs(src - cached).max())}"
    )


def test_ai_art_reference_texts_match_wama_csv():
    """REAP's `ai_art` cache `texts.json` matches the WAMA `chunk_text` column."""
    _require_sibling(WAMA_ROOT)
    with (WAMA_ROOT / "data/public_discourse_clean_chunks.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)
        src_texts = [row["chunk_text"] for row in reader]
    with (CACHE_ROOT / "ai_art/2026-04-14/texts.json").open(encoding="utf-8") as f:
        cached_texts = json.load(f)
    assert len(src_texts) == 1736
    assert src_texts == cached_texts


# --------------------------------------------------------------------------- #
# AI-art OOS: Lovato 2024 artist probes (1,259 × 1,024)                       #
# --------------------------------------------------------------------------- #


def test_ai_art_oos_artist_embeddings_match_wama_source():
    """REAP's `ai_art_oos_artist` embeddings are byte-identical to the WAMA NPY."""
    _require_sibling(WAMA_ROOT)
    src = np.load(
        WAMA_ROOT / "figures/prefix_grid_search/prefix_embeddings_artist.npy",
        allow_pickle=False,
    )
    cached = np.load(
        CACHE_ROOT / "ai_art_oos_artist/2026-05-21/embeddings.npy", allow_pickle=False
    )
    assert src.shape == cached.shape == (1259, 1024)
    assert src.dtype == cached.dtype == np.float32
    assert np.array_equal(src, cached), (
        f"max abs diff = {float(np.abs(src - cached).max())}"
    )


def test_ai_art_oos_artist_texts_and_labels_match_wama_csv():
    """REAP's `ai_art_oos_artist` texts + labels reflect WAMA's CSV exactly."""
    _require_sibling(WAMA_ROOT)
    from reap.datasets.ai_art_oos import THEMES

    with (WAMA_ROOT / "data/artist_perspectives.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)
        src_themes = [row["question_group"].strip() for row in reader]
    # Re-open to read perspective_text in the same order (csv readers are exhausted).
    with (WAMA_ROOT / "data/artist_perspectives.csv").open(
        encoding="utf-8", newline=""
    ) as f:
        reader = csv.DictReader(f)
        src_texts = [row["perspective_text"] for row in reader]

    with (CACHE_ROOT / "ai_art_oos_artist/2026-05-21/texts.json").open(
        encoding="utf-8"
    ) as f:
        cached_texts = json.load(f)
    cached_labels = np.load(
        CACHE_ROOT / "ai_art_oos_artist/2026-05-21/labels.npy", allow_pickle=False
    )

    assert len(src_texts) == 1259
    assert src_texts == cached_texts

    expected_labels = np.array(
        [THEMES.index(t) for t in src_themes], dtype=np.int64
    )
    assert np.array_equal(expected_labels, cached_labels)


# --------------------------------------------------------------------------- #
# AI-art OOS: style-controlled public probes (750 × 1,024)                    #
# --------------------------------------------------------------------------- #


def test_ai_art_oos_public_embeddings_match_wama_source():
    """REAP's `ai_art_oos_public` embeddings are byte-identical to the WAMA NPY."""
    _require_sibling(WAMA_ROOT)
    src = np.load(
        WAMA_ROOT / "figures/prefix_grid_search/prefix_embeddings_probes.npy",
        allow_pickle=False,
    )
    cached = np.load(
        CACHE_ROOT / "ai_art_oos_public/2026-05-21/embeddings.npy", allow_pickle=False
    )
    assert src.shape == cached.shape == (750, 1024)
    assert src.dtype == cached.dtype == np.float32
    assert np.array_equal(src, cached), (
        f"max abs diff = {float(np.abs(src - cached).max())}"
    )


def test_ai_art_oos_public_texts_and_labels_match_wama_csv():
    """REAP's `ai_art_oos_public` texts + labels reflect WAMA's CSV exactly."""
    _require_sibling(WAMA_ROOT)
    from reap.datasets.ai_art_oos import THEMES

    src_themes: list[str] = []
    src_texts: list[str] = []
    with (WAMA_ROOT / "data/public_probes.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src_themes.append(row["theme"].strip())
            src_texts.append(row["probe_text"])

    with (CACHE_ROOT / "ai_art_oos_public/2026-05-21/texts.json").open(
        encoding="utf-8"
    ) as f:
        cached_texts = json.load(f)
    cached_labels = np.load(
        CACHE_ROOT / "ai_art_oos_public/2026-05-21/labels.npy", allow_pickle=False
    )

    assert len(src_texts) == 750
    assert src_texts == cached_texts

    expected_labels = np.array(
        [THEMES.index(t) for t in src_themes], dtype=np.int64
    )
    assert np.array_equal(expected_labels, cached_labels)


# --------------------------------------------------------------------------- #
# Korean forest: reference (905 × 384)                                        #
# --------------------------------------------------------------------------- #


def test_korean_forest_reference_embeddings_match_hye_in_csv():
    """KF reference embeddings = float32 cast of the hye_in CSV (header skipped)."""
    _require_sibling(HYEIN_ROOT)
    src_path = HYEIN_ROOT / "data/og_data/Sections_of_Strategy_Sentences_embeddings6.csv"
    with src_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = [[float(x) for x in row] for row in reader]
    src = np.asarray(rows, dtype=np.float64).astype(np.float32, copy=False)

    cached = np.load(
        CACHE_ROOT / "korean_forest/2026-04-14/embeddings.npy", allow_pickle=False
    )
    assert src.shape == cached.shape == (905, 384)
    assert cached.dtype == np.float32
    assert np.array_equal(src, cached), (
        f"max abs diff = {float(np.abs(src - cached).max())}"
    )


def test_korean_forest_reference_texts_and_labels_match_hye_in_csv():
    """KF reference texts + sorted-unique strategy labels match the hye_in CSV."""
    _require_sibling(HYEIN_ROOT)
    import pandas as pd

    text_df = pd.read_csv(
        HYEIN_ROOT / "data/og_data/Sections_of_Strategy_Sentences6.csv",
        encoding="utf-8-sig",
    )
    assert len(text_df) == 905

    src_texts = text_df["text"].tolist()
    strategies = text_df["strategy"].tolist()
    unique_strategies = sorted(set(strategies))
    strategy_to_code = {s: i for i, s in enumerate(unique_strategies)}
    expected_labels = np.asarray(
        [strategy_to_code[s] for s in strategies], dtype=np.int64
    )

    with (CACHE_ROOT / "korean_forest/2026-04-14/texts.json").open(encoding="utf-8") as f:
        cached_texts = json.load(f)
    cached_labels = np.load(
        CACHE_ROOT / "korean_forest/2026-04-14/labels.npy", allow_pickle=False
    )
    assert src_texts == cached_texts
    assert np.array_equal(expected_labels, cached_labels)
    assert len(unique_strategies) == 20  # the documented K


# --------------------------------------------------------------------------- #
# Korean forest OOS: pre-projected pledges (1,662 × 18)                       #
# --------------------------------------------------------------------------- #


def test_korean_forest_oos_payload_matches_hye_in_xlsx():
    """KF OOS embeddings + cluster_labels + presidents + texts all match the 3 XLSX files."""
    _require_sibling(HYEIN_ROOT)
    pd = pytest.importorskip("pandas")
    pytest.importorskip("openpyxl")

    expected_per_president = {"Lee": 480, "Park": 633, "Moon": 549}
    dim_cols = [f"D{i}" for i in range(1, 19)]

    emb_rows = []
    cluster_rows = []
    president_rows = []
    texts: list[str] = []
    for president in ("Lee", "Park", "Moon"):
        df = pd.read_excel(
            HYEIN_ROOT / f"data_may7_2026/projections/{president}_projection.xlsx"
        )
        if "ㅎ" in df.columns and "text_sentence" not in df.columns:
            df = df.rename(columns={"ㅎ": "text_sentence"})
        assert len(df) == expected_per_president[president]
        emb_rows.append(df[dim_cols].to_numpy(dtype=np.float64))
        cluster_rows.append((df["cluster_k"].astype(int).to_numpy() - 1).astype(np.int64))
        president_rows.append(np.array([president] * len(df), dtype=object))
        texts.extend(str(t) for t in df["text_sentence"].tolist())

    src_emb = np.concatenate(emb_rows, axis=0)
    src_cluster = np.concatenate(cluster_rows, axis=0)
    src_president = np.concatenate(president_rows, axis=0)

    snap_dir = CACHE_ROOT / "korean_forest_oos/2026-05-09"
    cached_emb = np.load(snap_dir / "embeddings.npy", allow_pickle=False)
    cached_cluster = np.load(snap_dir / "cluster_labels.npy", allow_pickle=False)
    cached_president = np.load(snap_dir / "presidents.npy", allow_pickle=True)
    with (snap_dir / "texts.json").open(encoding="utf-8") as f:
        cached_texts = json.load(f)

    assert src_emb.shape == cached_emb.shape == (1662, 18)
    assert cached_emb.dtype == np.float64
    assert np.array_equal(src_emb, cached_emb), (
        f"max abs diff = {float(np.abs(src_emb - cached_emb).max())}"
    )
    assert np.array_equal(src_cluster, cached_cluster)
    assert np.array_equal(src_president, cached_president)
    assert texts == cached_texts
