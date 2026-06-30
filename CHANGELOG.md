# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog v1.1](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tracks work landing toward the first stable release (`1.0.0`).

### Added

- BERTopic baseline pipeline wrapper for like-for-like comparison against
  the REAP consensus pipeline on identical splits, seeds, and embeddings.
- Topic-coherence metrics (UMass, CV) computed from c-TF-IDF cluster terms
  for downstream topic-modeling evaluation.
- Conformal out-of-sample (OOS) filter for projection-head outputs,
  flagging points whose embedding falls outside the training-time
  conformal threshold.
- Neural projection head (`src/reap/projection.py`) for mapping new
  embeddings into the consensus space without re-running multi-seed UMAP.
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
