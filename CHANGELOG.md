# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog v1.1](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Tracks work landing toward the first stable release (`1.0.0`).

### Added

- Fail-closed hold-out aggregator (`scripts/aggregate_holdout_results.py`).
  It combines every in-scope cell under the protocol v1.6 aggregation
  rules: two-sided paired Wilcoxon tests by seed (replicates averaged per
  seed; identity tested against its single value), mean paired differences,
  paired Cohen's d_z, 95% seed-bootstrap intervals, and Holm within each
  registered family with Benjamini-Hochberg alongside. A missing, failed,
  uncommitted-code, or extra run stops it with the list of problems. An
  untestable comparison stays in its family with p = 1, so family sizes
  always equal the registered sizes. It also evaluates the acceptance
  expectations. `--status` writes the accounting table: every registered
  cell marked completed, failed, pending, or deferred with its reason.

- Hold-out cell driver (`scripts/run_holdout_cell.py`): runs one in-scope
  cell of `docs/run_matrix_v1_6.json` end to end. It builds the split as
  protocol v1.6 clause (e) fixes it, the in-set consensus from the 30 Set D
  seeds, K, and the adapted-ARI reference partition, then runs the hold-out
  harness, the subsample intervals, and the leakage audit. Every artifact
  and a `bundle.json` record go to the research root. It refuses deferred
  cells, protocols older than 1.6, code not yet on `origin/main` or with
  uncommitted changes, and output directories that already hold a completed
  result. A crash still leaves a failed record. `--preflight` reports split
  sizes, groups on both sides and leakage without training.

- Protocol v1.6 amendment draft (`docs/proposals/2026-11-protocol-v1.6-amendment.md`),
  written before any hold-out run. It runs 11 of the 23 registered hold-out
  evaluations first (AI-art T1–T5; Korean forest K1 and K2-T1…T5 on the
  canonical encoder) and defers the 12 that need embeddings not yet built,
  as their own multiple-comparison family; defines the 30 per-seed values in
  every paired test as 30 training runs per trained method against one
  consensus, with a separate resampling interval over held-out documents for
  generalization; and stops §18.h from accepting a cross-method win on
  distance correlation. `docs/run_matrix_v1_6.json` marks every cell in
  scope or deferred, and `tests/test_run_matrix_v1_6.py` recomputes the
  families from it. The v1.5 matrix is kept unchanged.
- Versioned record file for results (`src/reap/repro_bundle.py`, exported
  from `reap`). `ReproBundle` is a frozen, strict model of `bundle.json`:
  protocol version, code commit plus whether tracked files had uncommitted
  changes, script, run id, dataset and its SHA-256, seed set and the exact
  seeds, full settings, evaluation mode (`cv` / `final_fit` / `holdout`),
  metric recipe ids, the numerical environment (versions of numpy, scipy,
  scikit-learn, umap-learn, pynndescent, numba and torch, plus the BLAS and
  LAPACK libraries numpy was built with), and run status with a required
  failure reason for failed runs. Unknown, missing, or inconsistent fields
  fail on both write and read. `write_repro_bundle` writes atomically through
  a uniquely named temporary file. `validate_protocol_version` is the gate
  that drivers and the aggregation step use to refuse a protocol older than
  the one in force.
- Hold-out comparison harness (`reap.holdout`, main names exported from
  `reap`). For one hold-out split, `run_holdout_comparison` trains the REAP
  MLP head, the linear head and parametric UMAP once per training seed against
  the one in-set consensus, runs the identity baseline once, and scores every
  run on the four registered hold-out metrics with their recipes spelled out:
  trustworthiness and distance correlation between raw and placed held-out
  documents, silhouette under nearest-centroid labels, and the adapted ARI
  against a supplied reference partition. Held-out sets below the protocol's
  size floors get no numbers, and an undefined metric is recorded as missing
  with a reason, never as a sentinel value. `get_paired_subsample_intervals`
  gives the pre-designated generalization intervals.
- `reap.baselines.run_parametric_umap_holdout`: fits parametric UMAP on the
  in-set documents only and passes held-out documents through the fitted
  encoder, which the hold-out comparisons need.

- Identity hold-out baseline (`reap.baselines.get_identity_projection`): each
  new document takes the reference-map coordinate of its nearest reference
  document in the original embedding space (cosine by default, exact search,
  ties to the lowest index, no randomness). It is the pre-registered
  "just copy your nearest neighbour" floor for the hold-out evaluations, and
  its result also reports each new document's similarity to that neighbour,
  which the random-split leakage audit needs.
- Raw Korean-forest pledge embeddings (`reap.datasets.load_korean_forest_oos_raw`,
  built with `scripts/build_datasets.py --dataset korean_forest_oos_raw`). The
  existing pledge snapshot holds only the 18-d output of an earlier projection
  head; the corpus-level hold-out evaluation needs the 1,662 pledges as raw
  384-d MiniLM embeddings in the same space as the reference corpus. They are
  read as the source pipeline stored them, not re-embedded, row-aligned with
  `load_korean_forest_oos()` (the build refuses otherwise). Re-encoding the
  pledge text with the same model reproduces 1,661 of 1,662 vectors to float
  precision; that check and the difference in how reference and pledge text
  were preprocessed are recorded in the snapshot metadata.

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
- CI/packaging hardening batch: `src/reap/py.typed` now ships in the wheel
  so downstream type checkers consume the package's annotations; all five
  PR workflows run with least-privilege `permissions: contents: read` and
  job timeouts; the build and release workflows install the built wheel
  into a clean venv, import it, assert `py.typed` is inside, and assert the
  sdist carries no `manuscript/` or `results/` entries; the torch install
  hint names the real package (`reap-topics[projection]` — it pointed at a
  nonexistent `reap-embeddings`) with a test pinning it; the test matrix
  covers Python 3.10–3.13 (cp313 wheels for numba/torch verified on PyPI
  before adding the cell); the typecheck job installs the tensorflow-free
  `[ci]` extra, keeping torch and tensorflow apart in every job.
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
- Contribution-readiness files: `LICENSE` (MIT; see *Changed*), `CHANGELOG.md`,
  `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue and PR templates.
- `CITATION.cff` with the canonical citation metadata for all releases
  (author with ORCID and affiliation, MIT license, the Zenodo concept DOI,
  and the canonical title "REAP: Reproducible Embeddings via Averaged
  Projection"). The name expansion is now identical across the README,
  package metadata, package docstring, and repository description
  ("Embeddings", "Averaged" — two variants had drifted).

### Changed

- OpenAI defaults now follow one rule: `gpt-6.1-terra` for small to mid
  tasks and `gpt-6.1-sol` for difficult ones. Cluster labeling
  (`OPENAI_DEFAULT_MODEL`) and the standard topic-attribution judge use
  `gpt-6.1-terra`; the strong judge (`OPENAI_STRONG_MODEL`) uses
  `gpt-6.1-sol`. They replace `gpt-5.4-mini` and `gpt-5.5`. Labels produced
  earlier keep the model they were run with. The choice of OpenAI token
  parameter now names the legacy models (gpt-4*, gpt-3.5*) instead of the new
  ones, so newer model ids get `max_completion_tokens` without code changes.
- Tests and scripts no longer hardcode one machine's home-directory paths.
  Research artifacts are found in the REAP-research sibling first and this
  repo's `results/` second (`REAP_RESULTS_ROOT` overrides both); the AI-art and
  Korean-forest source projects come from `REAP_AI_ART_SOURCE_PATH` /
  `REAP_KOREAN_FOREST_SOURCE_PATH` or the sibling checkouts. The old paths had
  gone stale when the checkouts moved, so the real-corpus verification tests
  were silently skipping everywhere. They now run whenever a REAP-research
  checkout is present, where a missing artifact is a failure, and skip with a
  visible reason only on code-only checkouts such as CI. A new guard test fails
  on any home-directory path in `src/`, `scripts/`, `tests/`, or `examples/`.
- Relicensed from Apache-2.0 to MIT. Pre-release, single-author history, so
  the switch is clean; it settles the licensing note that listed MIT as a
  candidate pending confirmation. `LICENSE`, the `pyproject.toml` license
  field, and the trove classifier changed together.
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

[Unreleased]: https://github.com/Thru-Echoes/REAP/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Thru-Echoes/REAP/releases/tag/v0.1.0
