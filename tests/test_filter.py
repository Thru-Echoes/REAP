"""Tests for the OOS conformal filter (manuscript §3.6).

Covers:

- **Tier-1 mathematical invariants** — per-cluster threshold finite,
  retention monotone in α, calibration warning on small clusters,
  centroid/covariance shapes correct.
- **Korean forest validation** — confirms the implementation reproduces
  the sibling-project numbers from
  `green-narrative/hye_in/for_hyein/park_moon_results_2026-05-07_v2/`
  exactly (deterministic algorithm, no stochasticity tolerance):
    - Mahalanobis + pooled correction at α=0.01: 846/1662 = 50.9%
    - Mahalanobis + per-president correction at α=0.01: 868/1662 = 52.2%
    - Mahalanobis (no correction) at α=0.01: 447/1662 = 26.9%
    - Per-president and per-cluster breakdowns
- **REAP cache replication** — same headline retention via
  `reap.datasets.load_korean_forest()` + `load_korean_forest_oos()`,
  showing the package can replicate the sibling-project numbers from
  its own committed snapshots.
- **Synthetic-shift Tier-2 ranges** per evaluation_protocol.md §6.c —
  in-distribution retention ≥ 0.85, shifted retention ≤ 0.30.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reap.filter import (
    FilterCalibration,
    apply,
    calibrate,
    calibrate_per_subgroup,
)

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

# Sibling-project artifact paths. Tests that depend on these skip cleanly
# when the green-narrative project is not on disk (CI machines etc.).
_GN_REF_CSV = Path(
    "/Users/echoes/Documents/Berkeley/Research/green-narrative/hye_in/"
    "for_hyein/embeddings6_projected.csv"
)
_GN_OOS_DIR = Path(
    "/Users/echoes/Documents/Berkeley/Research/green-narrative/hye_in/"
    "data_may7_2026/projections"
)
_DIMS = [f"D{i}" for i in range(1, 19)]


def _have_korean_forest_oos_artifacts() -> bool:
    return _GN_REF_CSV.is_file() and all(
        (_GN_OOS_DIR / f"{p}_projection.xlsx").is_file()
        for p in ("Lee", "Park", "Moon")
    )


@pytest.fixture(scope="module")
def korean_forest_filter_inputs():
    """Load reference and OOS arrays from the green-narrative artifacts.

    Skips if the artifacts are not present on this machine.
    """
    if not _have_korean_forest_oos_artifacts():
        pytest.skip("green-narrative Korean forest OOS artifacts not on disk")
    pytest.importorskip("openpyxl")
    ref_df = pd.read_csv(_GN_REF_CSV)
    ref_coords = ref_df[_DIMS].values.astype(np.float64)
    ref_labels = ref_df["cluster_k"].astype(int).values  # already 0-indexed
    oos_dfs = []
    for p in ("Lee", "Park", "Moon"):
        df = pd.read_excel(_GN_OOS_DIR / f"{p}_projection.xlsx")
        df = df.rename(columns={"ㅎ": "text_sentence"})
        df = df.assign(_president=p)
        oos_dfs.append(df)
    oos_df = pd.concat(oos_dfs, ignore_index=True)
    oos_coords = oos_df[_DIMS].values.astype(np.float64)
    oos_labels = oos_df["cluster_k"].astype(int).values - 1  # normalize to 0-indexed
    oos_president = oos_df["_president"].values
    return ref_coords, ref_labels, oos_coords, oos_labels, oos_president


def _build_synthetic_fixture(
    *,
    n_clusters: int = 8,
    n_per_ref: int = 110,
    n_per_oos: int = 20,
    d: int = 6,
    center_scale: float = 8.0,
    ref_sigma: float = 1.0,
    oos_sigma: float = 1.0,
    seed_ref: int = 42,
    seed_oos: int = 43,
):
    """Build (reference, OOS) for an 8-cluster Gaussian fixture.

    Reference clusters share isotropic Gaussian shape with std ``ref_sigma``;
    OOS clusters share isotropic Gaussian shape with std ``oos_sigma``. When
    ``oos_sigma == ref_sigma`` the OOS is in-distribution; ``oos_sigma >>
    ref_sigma`` produces an off-shape shifted distribution where the filter
    should reject most points.
    """
    rng = np.random.default_rng(seed_ref)
    centers = rng.normal(scale=center_scale, size=(n_clusters, d))
    ref_coords = []
    ref_labels = []
    for c in range(n_clusters):
        ref_coords.append(centers[c] + rng.normal(scale=ref_sigma, size=(n_per_ref, d)))
        ref_labels.extend([c] * n_per_ref)
    rng2 = np.random.default_rng(seed_oos)
    oos_coords = []
    oos_labels = []
    for c in range(n_clusters):
        oos_coords.append(centers[c] + rng2.normal(scale=oos_sigma, size=(n_per_oos, d)))
        oos_labels.extend([c] * n_per_oos)
    return (
        np.vstack(ref_coords),
        np.asarray(ref_labels, dtype=int),
        np.vstack(oos_coords),
        np.asarray(oos_labels, dtype=int),
    )


@pytest.fixture
def synthetic_in_distribution():
    """8-cluster reference + matched in-distribution OOS (same shape)."""
    return _build_synthetic_fixture(oos_sigma=1.0)


@pytest.fixture
def synthetic_shifted():
    """8-cluster reference + off-shape OOS with 5x larger isotropic noise.

    The filter should reject the bulk of the OOS because the points sit far
    outside the reference covariance ellipsoid (Mahalanobis ≫ reference LOO
    99th percentile).
    """
    return _build_synthetic_fixture(oos_sigma=5.0)


# --------------------------------------------------------------------------
# Tier-1 invariants
# --------------------------------------------------------------------------


class TestCalibrateInvariants:
    """Mathematical invariants on the calibration output."""

    def test_returns_filter_calibration_instance(self, synthetic_in_distribution):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        assert isinstance(cal, FilterCalibration)

    def test_per_cluster_threshold_finite_for_each_present_cluster(
        self, synthetic_in_distribution
    ):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        for c in np.unique(ref_l):
            assert np.isfinite(cal.per_cluster_thresholds[int(c)]), (
                f"cluster {c} threshold should be finite (n_ref={int((ref_l == c).sum())})"
            )

    def test_per_cluster_threshold_is_non_negative(self, synthetic_in_distribution):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        for t in cal.per_cluster_thresholds.values():
            assert t >= 0.0

    def test_per_cluster_centroid_shape_matches_d(self, synthetic_in_distribution):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        d = ref.shape[1]
        for c, mu in cal.per_cluster_centroids.items():
            assert mu.shape == (d,), f"centroid for cluster {c} has wrong shape"

    def test_per_cluster_inv_covariance_shape_is_d_by_d(
        self, synthetic_in_distribution
    ):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        d = ref.shape[1]
        for ic in cal.per_cluster_inv_covariances.values():
            assert ic.shape == (d, d)

    def test_calibration_records_alpha_and_method(self, synthetic_in_distribution):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.025)
        assert cal.alpha == 0.025
        assert cal.method == "mahalanobis_pooled"

    def test_calibration_records_n_reference_and_oos(self, synthetic_in_distribution):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        assert cal.n_reference == len(ref)
        assert cal.n_oos == len(oos)


class TestApplyInvariants:
    """Mathematical invariants on filter application."""

    def test_apply_returns_bool_mask_of_correct_length(
        self, synthetic_in_distribution
    ):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l)
        keep = apply(oos, oos_l, cal)
        assert keep.shape == (len(oos),)
        assert keep.dtype == bool

    def test_retention_monotone_decreasing_in_alpha(self, synthetic_in_distribution):
        """As α decreases (stricter quantile), retention is non-decreasing.

        At α=0.10 the threshold is the 90th percentile (stricter; rejects more);
        at α=0.001 the threshold is the 99.9th percentile (looser; rejects fewer).
        Therefore retention(α=0.001) >= retention(α=0.01) >= retention(α=0.10).
        """
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        retentions = []
        for alpha in (0.10, 0.05, 0.01, 0.001):
            cal = calibrate(ref, ref_l, oos, oos_l, alpha=alpha)
            keep = apply(oos, oos_l, cal)
            retentions.append(int(keep.sum()))
        # Each successive value should be >= the previous (smaller alpha → looser)
        assert retentions == sorted(retentions), (
            f"retention should be monotone non-decreasing as α decreases: {retentions}"
        )


class TestCalibrationWarning:
    """Warning behavior for small reference clusters."""

    def test_warns_when_n_c_below_one_over_alpha(self):
        """A reference cluster with n_c < ⌈1/α⌉ gets a calibration warning.

        At α=0.01 the empirical 99th percentile of n_c=8 LOO values is just
        the maximum; the test that the (1-α) quantile is well-defined fails.
        """
        rng = np.random.default_rng(0)
        d = 4
        # 3 clusters: one with n=8 (small), two with n=30
        ref = np.vstack([
            rng.normal(loc=0, scale=1.0, size=(8, d)),
            rng.normal(loc=10, scale=1.0, size=(30, d)),
            rng.normal(loc=20, scale=1.0, size=(30, d)),
        ])
        ref_l = np.asarray([0] * 8 + [1] * 30 + [2] * 30, dtype=int)
        oos = rng.normal(loc=0, scale=1.0, size=(15, d))
        oos_l = np.array([0, 1, 2] * 5, dtype=int)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        small_warnings = [
            w for w in caught
            if "n_c" in str(w.message) or "small" in str(w.message).lower()
        ]
        assert len(small_warnings) >= 1, (
            f"expected a calibration warning for small cluster (n_c=8 < ⌈1/0.01⌉=100); "
            f"got: {[str(w.message) for w in caught]}"
        )


# --------------------------------------------------------------------------
# Korean forest validation — confirms exact sibling-project numbers
# --------------------------------------------------------------------------


class TestKoreanForestValidation:
    """Validate against `park_moon_results_2026-05-07_v2/` numbers exactly.

    The algorithm is deterministic given the inputs, so retention should
    match to the integer (not just the percent).
    """

    def test_pooled_overall_retention_matches_846_of_1662(
        self, korean_forest_filter_inputs
    ):
        ref, ref_l, oos, oos_l, _ = korean_forest_filter_inputs
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        keep = apply(oos, oos_l, cal)
        assert int(keep.sum()) == 846, (
            f"expected 846/1662; got {int(keep.sum())}/{len(oos)}"
        )

    def test_pooled_per_president_retention_matches(
        self, korean_forest_filter_inputs
    ):
        ref, ref_l, oos, oos_l, oos_p = korean_forest_filter_inputs
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        keep = apply(oos, oos_l, cal)
        # Lee 238/480, Park 320/633, Moon 288/549
        expected = {"Lee": 238, "Park": 320, "Moon": 288}
        for p, want in expected.items():
            mask = oos_p == p
            got = int(keep[mask].sum())
            assert got == want, f"{p}: expected {want}, got {got}"

    def test_pooled_per_cluster_retention_matches(self, korean_forest_filter_inputs):
        ref, ref_l, oos, oos_l, _ = korean_forest_filter_inputs
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        keep = apply(oos, oos_l, cal)
        # 0-indexed cluster -> (n_kept, n_total)
        expected = {
            0: (0, 2),
            1: (29, 33),
            2: (87, 288),
            3: (170, 269),
            4: (446, 724),
            5: (68, 218),
            6: (39, 118),
            7: (7, 10),
        }
        for c, (want_kept, want_total) in expected.items():
            mask = oos_l == c
            got_total = int(mask.sum())
            got_kept = int(keep[mask].sum())
            assert got_total == want_total, (
                f"cluster {c}: total mismatch — expected {want_total}, got {got_total}"
            )
            assert got_kept == want_kept, (
                f"cluster {c}: kept mismatch — expected {want_kept}, got {got_kept}"
            )

    def test_no_correction_overall_retention_matches_447_of_1662(
        self, korean_forest_filter_inputs
    ):
        ref, ref_l, oos, oos_l, _ = korean_forest_filter_inputs
        cal = calibrate(
            ref, ref_l, oos, oos_l, alpha=0.01, method="mahalanobis_no_correction"
        )
        keep = apply(oos, oos_l, cal)
        assert int(keep.sum()) == 447, (
            f"expected 447/1662 for no-correction variant; got {int(keep.sum())}"
        )

    def test_per_president_overall_retention_matches_868_of_1662(
        self, korean_forest_filter_inputs
    ):
        ref, ref_l, oos, oos_l, oos_p = korean_forest_filter_inputs
        cals = calibrate_per_subgroup(
            ref, ref_l, oos, oos_l, oos_p, alpha=0.01, fallback_n_threshold=10
        )
        # Apply per subgroup and concatenate
        keep = np.zeros(len(oos), dtype=bool)
        for sg, cal in cals.items():
            mask = oos_p == sg
            keep[mask] = apply(oos[mask], oos_l[mask], cal)
        assert int(keep.sum()) == 868, (
            f"expected 868/1662 for per-president variant; got {int(keep.sum())}"
        )

    def test_per_president_per_president_breakdown_matches(
        self, korean_forest_filter_inputs
    ):
        ref, ref_l, oos, oos_l, oos_p = korean_forest_filter_inputs
        cals = calibrate_per_subgroup(
            ref, ref_l, oos, oos_l, oos_p, alpha=0.01, fallback_n_threshold=10
        )
        keep = np.zeros(len(oos), dtype=bool)
        for sg, cal in cals.items():
            mask = oos_p == sg
            keep[mask] = apply(oos[mask], oos_l[mask], cal)
        expected = {"Lee": 247, "Park": 325, "Moon": 296}
        for p, want in expected.items():
            mask = oos_p == p
            got = int(keep[mask].sum())
            assert got == want, f"per-president variant {p}: expected {want}, got {got}"


# --------------------------------------------------------------------------
# Synthetic-shift Tier-2 golden ranges (evaluation_protocol.md §6.c)
# --------------------------------------------------------------------------


class TestKoreanForestViaPackageCache:
    """Same headline numbers must be reproducible from REAP's own cache.

    Skips when either the Korean forest reference or the OOS snapshot is
    missing from ``~/.cache/reap/datasets/``.
    """

    def _load_via_cache(self):
        from reap.datasets import load_korean_forest, load_korean_forest_oos

        try:
            ref = load_korean_forest()
        except FileNotFoundError as exc:
            pytest.skip(f"korean_forest cache missing: {exc}")
        try:
            oos = load_korean_forest_oos()
        except FileNotFoundError as exc:
            pytest.skip(f"korean_forest_oos cache missing: {exc}")
        return ref, oos

    def test_oos_snapshot_shape_matches_expected_1662_by_18(self):
        _, oos = self._load_via_cache()
        assert oos.embeddings.shape == (1662, 18)
        assert oos.cluster_labels.shape == (1662,)
        assert oos.presidents.shape == (1662,)
        assert len(oos.texts) == 1662

    def test_oos_snapshot_president_counts_match_expected(self):
        _, oos = self._load_via_cache()
        unique, counts = np.unique(oos.presidents, return_counts=True)
        seen = dict(zip(unique.tolist(), counts.tolist()))
        assert seen == {"Lee": 480, "Park": 633, "Moon": 549}

    def test_filter_via_cache_overall_retention_846_of_1662(
        self, korean_forest_filter_inputs
    ):
        _, oos = self._load_via_cache()
        # Reference 18-d coords are in the green-narrative artifact only
        # (REAP's load_korean_forest exposes raw 384-d MiniLM embeddings),
        # so cross-check by reusing the same green-narrative reference but
        # the cached OOS — verifying cache fidelity for the OOS payload.
        ref_coords = korean_forest_filter_inputs[0]
        ref_labels = korean_forest_filter_inputs[1]
        cal = calibrate(
            ref_coords,
            ref_labels,
            oos.embeddings.astype(np.float64),
            oos.cluster_labels.astype(int),
            alpha=0.01,
        )
        keep = apply(
            oos.embeddings.astype(np.float64),
            oos.cluster_labels.astype(int),
            cal,
        )
        assert int(keep.sum()) == 846


class TestSyntheticShiftGolden:
    """Pre-registered ranges from evaluation_protocol.md §6.c.

    in-distribution synthetic OOS retention ≥ 0.85
    shifted (off-shape) synthetic OOS retention ≤ 0.30
    """

    def test_in_distribution_retention_at_least_0_85(
        self, synthetic_in_distribution
    ):
        ref, ref_l, oos, oos_l = synthetic_in_distribution
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        keep = apply(oos, oos_l, cal)
        retention = float(keep.mean())
        assert retention >= 0.85, (
            f"in-distribution retention should be ≥ 0.85; got {retention:.3f}"
        )

    def test_shifted_retention_at_most_0_30(self, synthetic_shifted):
        ref, ref_l, oos, oos_l = synthetic_shifted
        cal = calibrate(ref, ref_l, oos, oos_l, alpha=0.01)
        keep = apply(oos, oos_l, cal)
        retention = float(keep.mean())
        assert retention <= 0.30, (
            f"shifted-distribution retention should be ≤ 0.30; got {retention:.3f}"
        )
