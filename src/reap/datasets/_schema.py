"""Pydantic schema for dataset snapshots consumed by REAP pipelines.

A `DatasetSnapshot` is the single contract between raw-data loaders and
the REAP algorithm surface. Every built-in dataset loader (AI-art,
Korean forest, etc.) and any bring-your-own-data path must produce a
`DatasetSnapshot` with fully-populated `DatasetMetadata`.

This module has no side effects.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetMetadata(BaseModel):
    """Provenance, licensing, and citation metadata for a dataset snapshot.

    Every field is required so that paper-adjacent artifacts produced from
    the snapshot can be traced back to their source without guesswork.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Canonical dataset name (e.g., 'ai_art_v1').",
    )
    version: str = Field(
        ...,
        min_length=1,
        description="Semantic version of this snapshot (e.g., '1.0.0').",
    )
    n_samples: int = Field(..., ge=1)
    embedding_dim: int = Field(..., ge=2)
    embedding_model: str = Field(
        ...,
        min_length=1,
        description=(
            "Embedding model identifier + version "
            "(e.g., 'intfloat/e5-large-v2 @ v1.0')."
        ),
    )
    preprocessing_version: str = Field(
        ...,
        min_length=1,
        description="Preprocessing pipeline version identifier.",
    )
    sha256: str = Field(
        ...,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="SHA256 digest (hex) of the snapshot's binary payload.",
    )
    license: str = Field(
        ...,
        min_length=1,
        description=(
            "License identifier for the corpus. SPDX identifier where the "
            "source provides one (e.g., 'CC-BY-4.0'); 'public-domain' for "
            "publicly-available corpora without a formal license; "
            "'NOASSERTION' when the licensing status is unknown."
        ),
    )
    citation: str | None = Field(
        default=None,
        description=(
            "BibTeX entry attributing the underlying corpus. May be None "
            "for pre-publication corpora; in that case `notes` should flag "
            "the citation as pending. Do not invent placeholders."
        ),
    )
    source_url: str | None = Field(
        default=None,
        description="Canonical source URL, if any.",
    )
    notes: str | None = Field(default=None)


class DatasetSnapshot(BaseModel):
    """Immutable bundle of embeddings, optional texts, and optional labels.

    Consumers should treat `embeddings` and `labels` as read-only even
    though numpy arrays are inherently mutable. Use `np.copy()` first if
    you need to mutate.

    `labels` are expert labels (for construct-validity comparison) — not
    pipeline output. Datasets without labels pass `labels=None`.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    embeddings: np.ndarray = Field(
        ...,
        description="(n_samples, embedding_dim) float array; finite, no NaN.",
    )
    texts: list[str] | None = Field(
        default=None,
        description="Source texts, one per sample, if available.",
    )
    labels: np.ndarray | None = Field(
        default=None,
        description="(n_samples,) integer expert labels, if available.",
    )
    metadata: DatasetMetadata

    @field_validator("embeddings")
    @classmethod
    def _embeddings_must_be_2d_finite(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"embeddings must be 2-d, got shape {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError(
                "embeddings contain NaN or Inf — clean data before wrapping"
            )
        return v

    @field_validator("labels")
    @classmethod
    def _labels_must_be_1d_int(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return v
        if v.ndim != 1:
            raise ValueError(f"labels must be 1-d, got shape {v.shape}")
        if not np.issubdtype(v.dtype, np.integer):
            raise ValueError(f"labels must be integer-typed, got dtype {v.dtype}")
        return v

    def model_post_init(self, _context: object) -> None:
        del _context  # pydantic passes context; REAP has no use for it.
        meta = self.metadata
        if self.embeddings.shape != (meta.n_samples, meta.embedding_dim):
            raise ValueError(
                f"embeddings shape {self.embeddings.shape} inconsistent with "
                f"metadata (n_samples={meta.n_samples}, "
                f"embedding_dim={meta.embedding_dim})"
            )
        if self.texts is not None and len(self.texts) != meta.n_samples:
            raise ValueError(
                f"texts length {len(self.texts)} != n_samples {meta.n_samples}"
            )
        if self.labels is not None and self.labels.shape != (meta.n_samples,):
            raise ValueError(
                f"labels shape {self.labels.shape} inconsistent with "
                f"n_samples={meta.n_samples}"
            )
