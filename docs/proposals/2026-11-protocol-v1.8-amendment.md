# Protocol v1.8 Amendment — Presidential Environmental Reference Map (Registered After Analysis)

**Status:** DRAFT — awaiting ratification. Ratification = merge of this
amendment's pull request by the project lead.
**Applies to:** `manuscript/evaluation_protocol.md` v1.7 → v1.8.
**Machine-readable companion:** `docs/run_matrix_v1_8.json`, checked by
`tests/test_run_matrix_v1_8.py`.

**Registered after analysis.** Unlike v1.5 to v1.7, this amendment is written
after the results it covers were computed and read. It does not claim
pre-registration. It records what was run, which settings were fixed while
results were being read, and the rules every reported number follows, so that
the case can be reported as final rather than exploratory. Section (e) lists
what this means for a reader.

The case is the third case study of the REAP thesis chapter: presidential
environmental language forms the reference map, and company environmental
language is placed on it. Its construction, runs and records live in the
research repository (REAP-research, branch `explore/env-reference-maps`,
`results/corporate_presidential/case3_rebuild/`). Paths below are relative to
that folder unless they name REAP.

---

## Clauses

### (a) Scope, and how this relates to v1.7

This amendment adds one tertiary corpus, the presidential environmental
reference map. It does not replace or change v1.7, which registers a
different corpus (earnings calls and presidential speech about climate and
energy, built from `cp-chunk-v1` passages and the `cp-keywords-v1` list).
Both stay registered.

### (b) The reference map

- **Units:** the 2,174 environmental units of the core presidential map
  `env_pres_unit_core_centred`: 1,459 of presidential speech and 715 of
  presidential written text (`placement/env_pres_unit_core_centred/placement_summary.json`,
  `map_passages_passing`). The sample and its selection are recorded in
  `speech_map/env_pres_unit_core_centred/sample_summary.json` and
  `selection.json`.
- **Encoder:** `intfloat/e5-large-v2`, revision
  `f169b11e22de13617baa190a028a32f3493550b6`, prefix `query: `, 1,024
  dimensions (`speech_map/env_pres_unit_core_centred/embedding.json`).
- **Centring:** each source group's average vector over the map sample
  (presidential speech, presidential written text) is removed from every
  passage of that group, and the result is rescaled to unit length
  (`speech_map/env_pres_unit_core_centred/centring.json`).
- **UMAP settings:** 18 dimensions, 30 neighbours, minimum distance 0, cosine
  input distance, taken from the recorded parameter search on exactly these
  vectors. REAP reads them from the search records
  `benchmark/search_records/reap.json` and `parametric_umap.json`, which name
  the input's SHA-256.

### (c) The placed company language

- **Placed sources:** company earnings calls (management prepared remarks,
  management answers, analyst questions) and company sustainability reports.
- **Rule:** a passage is placed when it is closer to the map than to the
  background by more than its source's threshold, in the cluster with the
  largest share of the neighbour vote (no minimum share). Closeness is the
  mean similarity to the five nearest passages of other documents. The rule,
  the per-source thresholds and the passages they were set on are fixed in
  `placement/env_pres_unit_core_centred/placement_summary.json`.
- **Placement method reported:** the neighbour vote. REAP's projection head
  is reported beside it, never instead of it; the head settings and the
  passages placed under each are in `head/env_pres_unit_core_centred/`
  (`projection.json`).

### (d) Seed sets, benchmark run and measures

- **Seeds:** REAP seed sets A, B and C, 30 seeds each, from REAP's
  `manuscript/seeds/seed_manifest.json` (SHA-256
  `88cefb0271b5a3c2ae6ac3df4fa3aa9bcf76d06b9a268ceea8526d869a462699`,
  unchanged between the run commit and v0.2.0).
- **Methods:** single seed, best of N, naive average, Procrustes, REAP and
  BERTopic through REAP's own `scripts/run_paper_benchmark.py`; parametric
  UMAP through the method-only entry of `scripts/run_bertopic_only.py`; the
  default-head check through `scripts/run_projection_head_oos.py`. The
  research driver `scripts/run_chapter2_case3_benchmark.py` registers the
  reference as one more dataset at run time and changes no REAP code. Every method reads the
  reference's searched settings, as for AI-art and Korean forest. The
  alternatives were not tuned separately.
- **K:** chosen per seed set by REAP's rule, the highest consensus silhouette
  for K from 5 to 30 (k-means seed 42). Fixed K of 12, the chapter's own cut,
  is a second reading. It never replaces the first and is never averaged with
  it. The chapter's 12 clusters were cut with the k-means seed at position 39
  of the research seed manifest; this is a separate cut from the benchmark's.
- **Headline measure:** REAP's mean between-set ARI over the three pairs of
  seed sets with K by the rule, reported with the three pair values and the K
  each set chose. AMI and NMI between sets, consensus silhouette and
  trustworthiness, and the default-head check (trustworthiness, ARI of a new
  k-means of predicted positions with the map clusters against the floor of
  0.5, distance correlation) are reported beside it.
- **No significance test** is run on the three pair values of a method.
- **Seed-set agreement and the two ten-seed search draws** are reported apart
  from the sets of 30 and are not compared with them.

These reading rules were committed in the research repository
(`docs/2026-11-25-chapter2-additions-plan.md`) before any benchmark number on
this case was computed.

### (e) What was fixed after analysis

These settings were chosen while results were being read, and then held fixed
for every reported number:

- the core map over the map of all units;
- centring by source;
- the 12-cluster cut;
- the neighbour vote over the projection head for placement;
- the thresholds of the background comparison.

The benchmark on this case was also added after the AI-art and Korean forest
benchmark results had been read. A reader should treat the case as a
demonstration on settings chosen with the data in view, not as a test of
settings fixed in advance.

### (f) Code version

The reported runs reproduce under REAP v0.2.0. Re-running seed-set
agreement, the six-method benchmark and parametric UMAP on seed sets A, B
and C at the v0.2.0 tag matched the earlier runs file for file, with no
difference in any label, coordinate or score (timing columns ignored); the
Korean harness check also reproduced exactly
(`benchmark/reap_release_check/`). The chapter cites REAP v0.2.0.

### (g) Open items, not registered here

- The 12 cluster labels have not yet been checked by a person.
- The placement check has one reader and covers only passages where the two
  placement rules disagree.

---

## Run matrix

`docs/run_matrix_v1_8.json` enumerates the runs: seed-set agreement (REAP,
three seed sets), the benchmark (six methods plus parametric UMAP, three seed
sets), and the default-head check (three seed sets). `tests/test_run_matrix_v1_8.py` checks
the counts and that every cell names a seed set, a method and a result path.
