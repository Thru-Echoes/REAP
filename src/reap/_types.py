"""Pydantic data models for the REAP pipeline.

All structured data crossing function boundaries uses these frozen models.
Models are organized by pipeline stage: configuration → results → analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class UmapParams(BaseModel):
    """UMAP hyperparameters for a single run."""

    model_config = ConfigDict(frozen=True)

    n_components: int = Field(..., ge=2, description="Target dimensionality")
    n_neighbors: int = Field(..., ge=2, description="Size of local neighborhood")
    min_dist: float = Field(
        ..., ge=0.0, le=1.0, description="Minimum distance between points (< 0.005 risks instability)"
    )
    metric: str = Field(default="cosine", description="Distance metric for high-d input space")
    random_state: int | None = Field(default=42, description="Random seed for reproducibility")


class KMeansParams(BaseModel):
    """KMeans clustering hyperparameters."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    n_clusters: int = Field(..., ge=2, alias="k", description="Number of clusters")
    init: str = Field(default="k-means++", description="Initialization method")
    n_init: int | str = Field(default="auto", description="Number of initializations")
    random_state: int | None = Field(default=42)


class ConsensusConfig(BaseModel):
    """Full REAP consensus pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    n_seeds: int = Field(..., ge=2, description="Number of UMAP seeds to run")
    umap: UmapParams
    kmeans: KMeansParams
    consensus_method: Literal["distance_matrix", "procrustes"] = Field(
        default="distance_matrix",
        description="distance_matrix (recommended, ARI +32%) or procrustes (baseline)",
    )


class ProjectionHeadConfig(BaseModel):
    """Neural projection head training configuration."""

    model_config = ConfigDict(frozen=True)

    input_dim: int = Field(default=384, description="Embedding dimension of the encoder")
    n_components: int = Field(default=18, description="Target consensus dimensionality")
    hidden_layers: list[int] = Field(
        default=[128, 64], description="Hidden layer sizes between input and output"
    )
    dropout: float = Field(default=0.3, ge=0.0, le=1.0)
    activation: str = Field(default="gelu", description="Activation function")
    loss_alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for MSE in combined loss (1-alpha = distance correlation weight)",
    )
    n_folds: int = Field(default=5, ge=2, description="Cross-validation folds")
    batch_size: int = Field(default=64, ge=1)
    max_epochs: int = Field(default=500, ge=1)
    patience: int = Field(default=50, ge=1, description="Early stopping patience")
    lr: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)


class GridSearchConfig(BaseModel):
    """Configuration for a parameter grid search."""

    model_config = ConfigDict(frozen=True)

    n_components_list: list[int] = Field(..., min_length=1)
    n_neighbors_list: list[int] = Field(..., min_length=1)
    min_dist_list: list[float] = Field(..., min_length=1)
    k_range: list[int] = Field(..., min_length=1, description="Range of K values to test")
    n_seeds: int = Field(default=30, ge=2)
    metric: str = Field(default="cosine")
    seeds: list[int] | None = Field(default=None, description="Explicit seed list (generated if None)")


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ConsensusResult(BaseModel):
    """Output from a single consensus UMAP run."""

    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(..., ge=1)
    n_dimensions: int = Field(..., ge=2)
    n_seeds: int = Field(..., ge=2)
    trustworthiness: float = Field(..., ge=0.0, le=1.0)
    silhouette: float = Field(..., ge=-1.0, le=1.0)
    best_k: int = Field(..., ge=2)
    mean_ari_seed_to_seed: float = Field(..., ge=-1.0, le=1.0)
    std_ari_seed_to_seed: float = Field(..., ge=0.0)
    mean_ari_seed_to_consensus: float = Field(..., ge=-1.0, le=1.0)
    std_ari_seed_to_consensus: float = Field(..., ge=0.0)
    config: ConsensusConfig


class GridSearchResult(BaseModel):
    """Metrics for a single grid search configuration."""

    model_config = ConfigDict(frozen=True)

    n_components: int
    n_neighbors: int
    min_dist: float
    trustworthiness: float = Field(..., ge=0.0, le=1.0)
    best_k: int = Field(..., ge=2)
    best_silhouette: float = Field(..., ge=-1.0, le=1.0)
    mean_ari_seeds: float
    std_ari_seeds: float
    mean_ari_consensus: float | None = None
    std_ari_consensus: float | None = None
    seed_k_mode: int | None = None
    seed_k_agree: int | None = None


class SilhouetteSearchResult(BaseModel):
    """Result of a K-selection via silhouette optimization."""

    model_config = ConfigDict(frozen=True)

    best_k: int = Field(..., ge=2)
    best_score: float = Field(..., ge=-1.0, le=1.0)
    scores: dict[int, float] = Field(..., description="K → silhouette score")


class ProjectionMetrics(BaseModel):
    """Evaluation metrics for the projection head."""

    model_config = ConfigDict(frozen=True)

    mse: float = Field(..., ge=0.0)
    trustworthiness: float = Field(..., ge=0.0, le=1.0)
    silhouette: float = Field(..., ge=-1.0, le=1.0)
    ari: float = Field(..., ge=-1.0, le=1.0)
    distance_correlation: float = Field(..., ge=-1.0, le=1.0)


class TrainingReport(BaseModel):
    """Complete projection head training report."""

    model_config = ConfigDict(frozen=True)

    cv_fold_metrics: list[ProjectionMetrics] = Field(..., min_length=1)
    cv_summary: dict[str, dict[str, float]]
    final_model_metrics: ProjectionMetrics
    config: ProjectionHeadConfig
    generated_at: datetime = Field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Labeling models
# ---------------------------------------------------------------------------


class CTfidfTerm(BaseModel):
    """A single term extracted via c-TF-IDF."""

    model_config = ConfigDict(frozen=True)

    term: str
    ngram_size: int = Field(..., ge=1)
    ctfidf_score: float = Field(..., ge=0.0)
    discriminativeness: float = Field(..., ge=0.0)
    cluster: int


class CTfidfClusterResult(BaseModel):
    """c-TF-IDF labeling result for one cluster."""

    model_config = ConfigDict(frozen=True)

    cluster: int
    n_items: int
    top_terms: list[CTfidfTerm]
    discriminative_terms: list[CTfidfTerm]
    shared_terms: list[CTfidfTerm]


class LLMClusterLabel(BaseModel):
    """LLM-generated label for one cluster."""

    model_config = ConfigDict(frozen=True)

    cluster: int
    label: str
    short_description: str
    evidence_summary: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    run_id: int = Field(default=0)


class PointStratification(BaseModel):
    """Classification of a point as core or peripheral."""

    model_config = ConfigDict(frozen=True)

    point_index: int
    cluster: int
    silhouette_score: float = Field(..., ge=-1.0, le=1.0)
    centroid_distance: float = Field(..., ge=0.0)
    stratum: Literal["core", "peripheral"]


class StratificationSummary(BaseModel):
    """Aggregate statistics from point stratification."""

    model_config = ConfigDict(frozen=True)

    silhouette_threshold: float
    total_points: int
    core_points: int
    peripheral_points: int
    core_silhouette_mean: float
    all_silhouette_mean: float
    core_cluster_sizes: dict[int, int]
    smallest_core_cluster: int


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


class RandomSeedSet(BaseModel):
    """Reproducible set of random seeds."""

    model_config = ConfigDict(frozen=True)

    master_seed: int
    n_seeds: int
    seeds: list[int]
    generated_at: datetime = Field(default_factory=datetime.now)

    def model_post_init(self, __context: object) -> None:
        if len(self.seeds) != self.n_seeds:
            raise ValueError(f"Expected {self.n_seeds} seeds, got {len(self.seeds)}")
