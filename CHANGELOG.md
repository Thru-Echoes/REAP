# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog v1.1](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tracks work landing toward the first stable release (`1.0.0`).

### Added

- Protocol v1.5 amendment draft (`docs/proposals/2026-07-protocol-v1.5-amendment.md`)
  locking the temporal-holdout scoring rules before any experiment runs:
  Holm multiple-comparison families defined once (AI-art 60 / Korean forest
  216 / core 5 / demographics 9) with the full run enumeration committed as
  `docs/run_matrix_v1_5.json` and recomputed in CI (`tests/test_run_matrix.py`);
  corrected acceptance denominators; trustworthiness designated the primary
  cross-method holdout metric; identity baseline reported as a primary
  comparison; leakage and repeated-measures rules; record-file schema
  additions with a mechanical protocol-version gate; deferred analyses and
  residual risks recorded. Ratification happens at the amendment PR's merge.
- Distance-correlation recipe pinning: the reported metric
  (`compute_distance_correlation`, Pearson r between condensed unique-pair
  distance vectors) and the training-loss term
  (`compute_projection_loss`, Pearson r over full flattened self-distance
  matrices, diagonal included) are two different calculations that shared
  one everyday name — and neither is Székely's distance correlation. They
  are now registered as distinct recipes (`distance_correlation`,
  `dist_corr_loss`) with from-scratch reference implementations, closed-form
  and synthetic verification rungs pinning each recipe and their exact gap,
  a byte-identical loss regression test (the loss numerics are
  calibration-frozen), and docstrings that name each recipe and
  cross-reference the other.
- Metrics catalog (`src/reap/metrics_catalog.py`): frozen Pydantic models
  binding every recorded metric value to a named recipe (the exact
  calculation — variant, label pairing, cluster-count rule, parameters), its
  provenance (source artifact, code commit, how strongly it is backed), and
  its evaluation mode; paired method-vs-method comparisons carry their test,
  corrected p-values, and effect sizes. Catalogs validate fail-closed
  (duplicate or dangling recipe ids refuse to load) and write atomically.
  Exported from the top-level `reap` package.
- BERTopic baseline pipeline wrapper for like-for-like comparison against
  the REAP consensus pipeline on identical splits, seeds, and embeddings.
- Topic-coherence metrics (UMass, CV) computed from c-TF-IDF cluster terms
  for downstream topic-modeling evaluation.
- Conformal out-of-sample (OOS) filter for projection-head outputs,
  flagging points whose embedding falls outside the training-time
  conformal threshold.
- Neural projection head (`src/reap/projection.py`) for mapping new
  embeddings into the consensus space without re-running multi-seed UMAP,
  with an MLP head and a `LinearProjectionHead` (the pre-registered linear
  baseline). Both heads, plus `train_projection_head` and
  `compute_projection_loss`, are exported from the top-level `reap` package.
- First-class dataset loaders under `src/reap/datasets/`:
  `load_ai_art()`, `load_korean_forest()`, `load_korean_forest_oos()`,
  with `DatasetSnapshot` / `DatasetMetadata` Pydantic v2 carriers.
- Manuscript-companion CI: `golden-validation` workflow guarding Tier-1
  (mathematical invariants), Tier-2 (statistical properties), and
  Tier-3 (tolerance snapshots) checks tied to the pre-registered
  evaluation protocol.
- GitHub Actions workflows for `test`, `typecheck`, `lint`, `build`, and a
  `release` workflow (build + verify only — PyPI publishing is intentionally
  disabled until REAP is publishable; see *Changed* below).
- Contribution-readiness files: `LICENSE` (Apache-2.0), `CHANGELOG.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and PR templates.

### Changed

- Removed the unverified "+32% ARI (0.75 vs. 0.56)" headline from the README
  and the `consensus.py` docstrings. The figure came from a sibling analysis
  under a definitionally different ARI calculation, and the README had
  additionally attributed it to the wrong corpus; quantitative claims now
  defer to committed benchmark artifacts. The tracked correction ledger
  (`docs/PENDING_PROSE_CORRECTIONS.md`) records the frozen-manuscript
  occurrences to fix at unfreeze.
- `train_projection_head` is reproducible by default: it takes an explicit
  `seed` (default 42) that controls weight initialization, per-fold shuffling,
  and the stratified fold splits. Pass `seed=None` to defer to a caller-set
  global torch seed instead — used by the calibrated golden fixtures and the
  out-of-sample scripts, which seed torch themselves.
- README reorganized: pitch -> install -> minimal working example ->
  links, with the longer API reference kept near the bottom.
- Disabled PyPI publication until REAP is a publishable package. Added the
  `Private :: Do Not Upload` classifier so the PyPI server rejects any upload
  from any source (CI or local `twine upload`), and removed the publish step,
  the OIDC `id-token: write` permission, and the `pypi` environment from the
  `release` workflow — which now builds and verifies the artifact but cannot
  publish. Re-enabling publication is a deliberate, reviewed change.

### Notes

- The version stays at `0.1.0` for this pre-release window; the next
  bump is gated on the manuscript's evaluation-protocol re-run and
  the Phase 6 release checklist.

## [0.1.0] — initial pre-release

Initial REAP package skeleton, intentionally narrow in scope. Captures
the state at first import-able snapshot.

### Added

- Core consensus pipeline: `run_consensus_pipeline`,
  `get_multi_seed_embeddings`, `get_consensus_distance_matrix`,
  `get_consensus_embedding`.
- Evaluation primitives: `compute_trustworthiness`, `compute_silhouette`,
  `compute_pairwise_ari`, `compute_seed_stability`.
- Clustering helpers: `find_best_k`, `run_kmeans`.
- Validation helpers: `validate_embeddings`, `validate_cluster_sizes`.
- Initial test suite under `tests/` covering consensus invariants,
  clustering, evaluation, and end-to-end smoke tests on synthetic blobs.
- Pydantic v2 data models for configuration and result carriers.
- `pyproject.toml` declaring runtime deps (numpy, scipy, scikit-learn,
  umap-learn, pydantic) and optional extras for `projection`,
  `labeling`, `baselines`, `text-fixtures`, `coherence`, and `dev`.

[Unreleased]: https://github.com/Thru-Echoes/reap-topics/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Thru-Echoes/reap-topics/releases/tag/v0.1.0
