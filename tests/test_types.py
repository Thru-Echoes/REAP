"""Tests for Pydantic data models."""

import pytest

from reap._types import (
    ConsensusConfig,
    GridSearchConfig,
    KMeansParams,
    ProjectionHeadConfig,
    RandomSeedSet,
    UmapParams,
)


class TestUmapParams:
    def test_valid_creation(self) -> None:
        p = UmapParams(n_components=18, n_neighbors=19, min_dist=0.005)
        assert p.n_components == 18
        assert p.metric == "cosine"

    def test_frozen(self) -> None:
        p = UmapParams(n_components=18, n_neighbors=19, min_dist=0.005)
        with pytest.raises(Exception):
            p.n_components = 10  # type: ignore[misc]

    def test_rejects_invalid_min_dist(self) -> None:
        with pytest.raises(Exception):
            UmapParams(n_components=18, n_neighbors=19, min_dist=1.5)


class TestConsensusConfig:
    def test_full_config(self) -> None:
        cfg = ConsensusConfig(
            n_seeds=30,
            umap=UmapParams(n_components=18, n_neighbors=19, min_dist=0.005),
            kmeans=KMeansParams(k=8),
        )
        assert cfg.consensus_method == "distance_matrix"
        assert cfg.umap.n_components == 18
        assert cfg.kmeans.n_clusters == 8


class TestRandomSeedSet:
    def test_valid(self) -> None:
        rs = RandomSeedSet(master_seed=42, n_seeds=3, seeds=[1, 2, 3])
        assert rs.n_seeds == 3

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected 3 seeds"):
            RandomSeedSet(master_seed=42, n_seeds=3, seeds=[1, 2])


class TestGridSearchConfig:
    def test_valid(self) -> None:
        cfg = GridSearchConfig(
            n_components_list=[5, 10, 18],
            n_neighbors_list=[15, 19, 27],
            min_dist_list=[0.005, 0.01, 0.1],
            k_range=[4, 6, 8, 10],
        )
        assert cfg.n_seeds == 30


class TestProjectionHeadConfig:
    def test_defaults(self) -> None:
        cfg = ProjectionHeadConfig()
        assert cfg.input_dim == 384
        assert cfg.n_components == 18
        assert cfg.dropout == 0.3
