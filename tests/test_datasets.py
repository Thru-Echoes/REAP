"""E2E tests for the REAP dataset loaders.

These tests read from the on-disk REAP cache. When the cache is empty
(CI machines without source projects), individual tests skip with a
pointer to ``scripts/build_datasets.py``. Locally on Oliver's machine
— where both sibling projects exist — the cache is populated by
running that builder once and every test runs.

The cache-building path itself is exercised by tests that point at a
temporary ``cache_dir`` and the environment variable
``REAP_AI_ART_SOURCE_PATH`` / ``REAP_KOREAN_FOREST_SOURCE_PATH``. When
those variables are unset, the source-project builders skip.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from reap.consensus import run_consensus_pipeline
from reap.datasets import (
    DatasetMetadata,
    DatasetSnapshot,
    build_ai_art_snapshot,
    build_korean_forest_snapshot,
    compute_payload_sha256,
    load_ai_art,
    load_korean_forest,
    load_snapshot,
    write_snapshot,
)
from reap.datasets.ai_art import (
    DATASET_VERSION as AI_ART_VERSION,
)
from reap.datasets.ai_art import (
    EXPECTED_EMBEDDING_DIM as AI_ART_DIM,
)
from reap.datasets.ai_art import (
    EXPECTED_N_SAMPLES as AI_ART_N,
)
from reap.datasets.korean_forest import (
    DATASET_VERSION as KF_VERSION,
)
from reap.datasets.korean_forest import (
    EXPECTED_EMBEDDING_DIM as KF_DIM,
)
from reap.datasets.korean_forest import (
    EXPECTED_N_SAMPLES as KF_N,
)


def _skip_if_cache_empty(loader, label: str):
    try:
        return loader()
    except FileNotFoundError as exc:
        pytest.skip(f"{label} cache not populated: {exc}")


# ---------------------------------------------------------------------------
# Cache-level roundtrip (always runs; no source projects required)
# ---------------------------------------------------------------------------


def test_write_and_load_snapshot_roundtrip(tmp_path: Path) -> None:
    """A hand-built snapshot round-trips through the cache intact."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=20).astype(np.int64)
    texts = [f"doc {i}" for i in range(20)]

    sha = compute_payload_sha256(X, y, texts)
    meta = DatasetMetadata(
        name="roundtrip_fixture",
        version="2026-04-14",
        n_samples=20,
        embedding_dim=8,
        embedding_model="test-encoder",
        preprocessing_version="test-prep",
        sha256=sha,
        license="CC0-1.0",
        citation="@misc{test,title={test},year={2026}}",
    )
    snap = DatasetSnapshot(embeddings=X, labels=y, texts=texts, metadata=meta)

    snap_dir = write_snapshot(snap, cache_dir=tmp_path)
    assert snap_dir.is_dir()
    assert (snap_dir / "embeddings.npy").is_file()
    assert (snap_dir / "labels.npy").is_file()
    assert (snap_dir / "texts.json").is_file()
    assert (snap_dir / "metadata.json").is_file()

    restored = load_snapshot("roundtrip_fixture", "2026-04-14", cache_dir=tmp_path)
    np.testing.assert_array_equal(restored.embeddings, X)
    assert restored.labels is not None
    np.testing.assert_array_equal(restored.labels, y)
    assert restored.texts == texts
    assert restored.metadata.sha256 == sha


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    """Corrupting the embeddings on disk surfaces as a ValueError at load."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((12, 4)).astype(np.float32)
    sha = compute_payload_sha256(X, None, None)
    meta = DatasetMetadata(
        name="tamper_fixture",
        version="2026-04-14",
        n_samples=12,
        embedding_dim=4,
        embedding_model="test",
        preprocessing_version="test",
        sha256=sha,
        license="CC0-1.0",
        citation="@misc{x,title={x},year={2026}}",
    )
    snap_dir = write_snapshot(
        DatasetSnapshot(embeddings=X, metadata=meta),
        cache_dir=tmp_path,
    )
    # Tamper with the embeddings on disk.
    tampered = X + 0.01
    np.save(snap_dir / "embeddings.npy", tampered, allow_pickle=False)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_snapshot("tamper_fixture", "2026-04-14", cache_dir=tmp_path)


def test_load_snapshot_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="REAP snapshot not found"):
        load_snapshot("does_not_exist", "2026-04-14", cache_dir=tmp_path)


def test_snapshot_labels_optional_roundtrip(tmp_path: Path) -> None:
    """A snapshot with `labels=None` round-trips without a labels.npy file."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((10, 4)).astype(np.float32)
    sha = compute_payload_sha256(X, None, None)
    meta = DatasetMetadata(
        name="no_labels_fixture",
        version="2026-04-14",
        n_samples=10,
        embedding_dim=4,
        embedding_model="test",
        preprocessing_version="test",
        sha256=sha,
        license="CC0-1.0",
        citation="@misc{x,title={x},year={2026}}",
    )
    snap_dir = write_snapshot(DatasetSnapshot(embeddings=X, metadata=meta), cache_dir=tmp_path)
    assert not (snap_dir / "labels.npy").exists()
    assert not (snap_dir / "texts.json").exists()

    restored = load_snapshot("no_labels_fixture", "2026-04-14", cache_dir=tmp_path)
    assert restored.labels is None
    assert restored.texts is None


# ---------------------------------------------------------------------------
# AI-art: runs only when the default cache is populated
# ---------------------------------------------------------------------------


def test_ai_art_snapshot_shape_and_metadata() -> None:
    snap = _skip_if_cache_empty(load_ai_art, "ai_art")
    assert snap.embeddings.shape == (AI_ART_N, AI_ART_DIM)
    assert snap.embeddings.dtype == np.float32
    assert np.isfinite(snap.embeddings).all()
    assert snap.labels is None
    assert snap.texts is not None
    assert len(snap.texts) == AI_ART_N
    assert snap.metadata.name == "ai_art"
    assert snap.metadata.version == AI_ART_VERSION
    assert snap.metadata.n_samples == AI_ART_N
    assert snap.metadata.embedding_dim == AI_ART_DIM
    assert "e5-large-v2" in snap.metadata.embedding_model


def test_ai_art_runs_through_consensus_pipeline() -> None:
    """The snapshot hands off cleanly to `run_consensus_pipeline`."""
    snap = _skip_if_cache_empty(load_ai_art, "ai_art")
    # 3 seeds, d=5 — small enough for fast CI while still exercising the
    # full consensus path on the real AI-art embeddings.
    result = run_consensus_pipeline(
        X=snap.embeddings,
        seeds=[0, 1, 2],
        n_components=5,
        n_neighbors=15,
        min_dist=0.01,
    )
    # run_consensus_pipeline returns (embedding, metadata)
    embedding, _metadata = result
    assert embedding.shape == (AI_ART_N, 5)
    assert np.isfinite(embedding).all()


# ---------------------------------------------------------------------------
# Korean forest: runs only when the default cache is populated
# ---------------------------------------------------------------------------


def test_korean_forest_snapshot_shape_and_metadata() -> None:
    snap = _skip_if_cache_empty(load_korean_forest, "korean_forest")
    assert snap.embeddings.shape == (KF_N, KF_DIM)
    assert snap.embeddings.dtype == np.float32
    assert np.isfinite(snap.embeddings).all()
    assert snap.labels is not None
    assert snap.labels.shape == (KF_N,)
    assert snap.texts is not None
    assert len(snap.texts) == KF_N
    assert snap.metadata.name == "korean_forest"
    assert snap.metadata.version == KF_VERSION
    assert snap.metadata.n_samples == KF_N
    assert "MiniLM" in snap.metadata.embedding_model


def test_korean_forest_expert_labels_are_20_strategies() -> None:
    """The expert taxonomy has 20 policy-strategy classes; mapping is recorded."""
    snap = _skip_if_cache_empty(load_korean_forest, "korean_forest")
    assert snap.labels is not None
    assert snap.metadata.notes is not None
    notes = json.loads(snap.metadata.notes)
    assert notes["n_classes"] == 20
    assert len(notes["strategy_label_map"]) == 20
    unique_labels = set(snap.labels.tolist())
    assert len(unique_labels) == 20


def test_korean_forest_runs_through_consensus_pipeline() -> None:
    snap = _skip_if_cache_empty(load_korean_forest, "korean_forest")
    result = run_consensus_pipeline(
        X=snap.embeddings,
        seeds=[0, 1, 2],
        n_components=5,
        n_neighbors=15,
        min_dist=0.01,
    )
    embedding, _metadata = result
    assert embedding.shape == (KF_N, 5)
    assert np.isfinite(embedding).all()


# ---------------------------------------------------------------------------
# Builder paths: auto-discover the sibling source projects
# ---------------------------------------------------------------------------


def _resolve_source_path(env_var: str, fallbacks: list[Path]) -> Path | None:
    """Find a sibling source-project root, with env-var override.

    Resolution order:
    1. ``$<env_var>`` if set and existing.
    2. The first ``fallback`` that exists on disk.
    3. ``None`` (caller skips with a clear message).
    """
    env = os.environ.get(env_var)
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p
    for p in fallbacks:
        if p.is_dir():
            return p
    return None


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SIBLING_ROOT = _REPO_ROOT.parent

_AI_ART_FALLBACKS = [
    _SIBLING_ROOT / "When-Algorithms-Meet-Artists",
    Path.home() / "Documents/Berkeley/Research/When-Algorithms-Meet-Artists",
]
_KOREAN_FOREST_FALLBACKS = [
    _SIBLING_ROOT / "green-narrative" / "hye_in",
    Path.home() / "Documents/Berkeley/Research/green-narrative/hye_in",
]


def test_build_ai_art_snapshot_roundtrip(tmp_path: Path) -> None:
    src = _resolve_source_path("REAP_AI_ART_SOURCE_PATH", _AI_ART_FALLBACKS)
    if src is None:
        pytest.skip(
            "AI-art source project not found. Set REAP_AI_ART_SOURCE_PATH or "
            f"check one of: {_AI_ART_FALLBACKS}"
        )
    snap_dir = build_ai_art_snapshot(src, cache_dir=tmp_path)
    assert snap_dir.is_dir()
    snap = load_snapshot("ai_art", AI_ART_VERSION, cache_dir=tmp_path)
    assert snap.embeddings.shape == (AI_ART_N, AI_ART_DIM)


def test_build_korean_forest_snapshot_roundtrip(tmp_path: Path) -> None:
    src = _resolve_source_path(
        "REAP_KOREAN_FOREST_SOURCE_PATH", _KOREAN_FOREST_FALLBACKS
    )
    if src is None:
        pytest.skip(
            "Korean forest source project not found. Set "
            "REAP_KOREAN_FOREST_SOURCE_PATH or check one of: "
            f"{_KOREAN_FOREST_FALLBACKS}"
        )
    snap_dir = build_korean_forest_snapshot(src, cache_dir=tmp_path)
    assert snap_dir.is_dir()
    snap = load_snapshot("korean_forest", KF_VERSION, cache_dir=tmp_path)
    assert snap.embeddings.shape == (KF_N, KF_DIM)
    assert snap.labels is not None
