# Protocol v1.5 Amendment — Temporal-Holdout Scoring Rules (Pre-Registration)

**Status:** DRAFT — awaiting ratification. Ratification = merge of this
amendment's pull request, by the project lead, at the start of Phase 2.
**Applies to:** `manuscript/evaluation_protocol.md` v1.4 → v1.5.
**Hard gate (unchanged):** no Phase 3/4/5 experiment runs before this
amendment lands. The record-file schema and experiment drivers enforce the
gate mechanically (`protocol_version >= 1.5` refusal — see clause k).
**Machine-readable companion:** `docs/run_matrix_v1_5.json` — the full
enumeration of the temporal-holdout program. `tests/test_run_matrix.py`
recomputes the family sizes stated here from that file and fails CI if they
drift apart.
**Sequencing note:** clause (m) cites the distance-correlation recipe
wording pinned by the recipe-pinning change (REAP PR #9); that PR merges
before this one.

This amendment closes the under-specified scoring details in §18/§19 before
any result exists, keeping the "report, don't gate" framing: every
enumerated comparison is reported; the pre-registered expectations below
feed specific claims but do not decide paper viability (clause j gives the
null path). Each clause is written to be pasted into (or referenced from)
the protocol; §15 gets one changelog entry covering all of them.

---

## Clauses

### (a) Per-corpus acceptance counts — replaces "3 of 6"

§18.h's "at least 3 of 6 strategies" matches neither corpus (AI-art runs 5
strategies; Korean forest runs 6 strategies × 3 encoders = 18 evaluations).
Corrected, preserving the original half-the-runs intent:

- **AI-art:** at least **3 of 5** strategies achieve trustworthiness
  mean > 0.70 AND distance-correlation mean > 0.70 on held-out.
- **Korean forest:** at least **9 of 18** evaluations achieve the same.
- K1 corpus-OOS trustworthiness > 0.65 stays as written.

These are pre-registered expectations that back the "REAP generalizes
out-of-sample" claim; failing them triggers clause (j), not paper death.

### (b) The "CV mean" qualifier is restored

Every number produced under cross-validation is labeled **"CV mean"**
wherever it appears (tables, prose, README), and the record file's
`evaluation_mode` field (clause k) must match the label
(`cv` / `final_fit` / `holdout`). A CV number quoted without the qualifier
is a correction-ledger entry.

### (c) Metric roles: trustworthiness primary, distance correlation supporting

**Trustworthiness is the primary cross-method metric** for the holdout
program: it is training-independent for every method compared.
Distance-correlation cross-method comparisons are labeled
**"training-objective-aligned, supporting"** — REAP's projection heads are
trained to maximize (a variant of) this quantity and the Parametric-UMAP
baseline is not, so a cross-method win on it is the first thing a methods
reviewer strikes. Head-vs-head comparisons on distance correlation
(MLP vs linear head) remain **primary** — both heads train on the same
loss, so the comparison is fair. Role marks are recorded per metric in the
run matrix; roles do NOT change family membership (clause d).

### (d) The Holm family, defined exactly once

Three inconsistent definitions existed (§8: ~5 per dataset per metric;
§18.g: "216 total per corpus"; the v1.4 proposal: 108). Resolution — the
project has **four pre-registered families**, each Holm-corrected
internally at α = 0.05, with BH-FDR reported alongside as sensitivity:

1. **Core methods family** (§3/§8, unchanged): per dataset per metric,
   REAP vs each of the five §3 baselines (single_seed, best_of_n,
   naive_average, procrustes, bertopic).
2. **AI-art holdout family**: 5 strategies × 3 paired baselines
   (parametric_umap, identity, linear_head) × 4 scalar test metrics
   (trustworthiness, distance_correlation, silhouette_holdout,
   ari_adapted_stage10).
3. **Korean-forest holdout family**: 18 evaluations × 3 × 4.
4. **A1 demographics family**: the §19.a chi-square screen (clause l).

Stated counts (machine-checked against `docs/run_matrix_v1_5.json` by
`tests/test_run_matrix.py`):

```
family_size(core_methods_per_dataset_per_metric) = 5
family_size(ai_art_holdout) = 60
family_size(korean_forest_holdout) = 216
family_size(a1_demographics) = 9
```

The old "216 total per corpus" was correct only for Korean forest
(18 × 3 × 4) and is retired as a corpus-generic count; the proposal's 108
derived from a superseded design and is retired. **Replicate rule:** the
random-split strategies (T5's three replicates; K2-T4; K2-T5) aggregate
each metric to the per-seed **mean across replicates** before any paired
test, so each strategy contributes exactly one 30-value column and the
counts above are exact. All enumerated tests are family members regardless
of primary/supporting role.

### (e) Identity baseline: reported, prominently, without a gate

The identity baseline (copy the nearest reference chunk's answer) is a
member of every holdout comparison cell. REAP-vs-identity is reported as a
**primary comparison** — effect size and CI, per corpus — with no gate.
Pre-written null interpretation: *if REAP does not beat identity on
trustworthiness for a corpus, the projection head adds nothing over
copying the nearest known point there; we say so plainly. That is a
negative finding for the projection component only — the consensus-space
contribution is assessed separately by the core §3/§6 comparisons.*

### (f) Dependence unit and leakage audit for random holdouts

AI-art rows are ~250-word chunks of longer documents; T5 stratifies by
year only, so chunks of one document can land on both sides of a split and
inflate every OOS metric. Pre-registered handling:

- If document identifiers are recoverable from the dataset snapshot,
  **T5 becomes group-aware**: all chunks of a document stay on one side.
- Otherwise, T5 results carry a **leakage audit**: the distribution of
  each held-out chunk's maximum cosine similarity to any in-set chunk,
  and the share above 0.95, reported next to the metrics; T5 is labeled
  "optimistic under chunk-level leakage".
- Either way, the **temporal strategies (T1–T4) are the primary
  generalization evidence**; the same rule applies to K2-T4/K2-T5 with
  speech/document as the grouping unit.
- §11 gains the objection-and-response row: *"Your random holdouts leak
  same-document chunks"* → group-aware splits where ids exist; reported
  leakage audit + temporal-primary designation where they do not.

### (g) Artist repeated-measures correction (§19.a) — §13 deviation linkage

A1's contingency tests treat the 1,259 artist probes as independent, but
probes are nested within respondents (several probes per artist).
Corrected primary analysis, pre-registered before any A1 number is
committed: **artist-level permutation test** (permute respondent-level
demographic labels; recompute chi-square) or artist-level aggregation
(one vector per respondent) — the driver implements the permutation test
as primary. **Bootstrap CIs resample artists, not probes.** Existing
per-probe results exist only as local, untracked artifacts; they are
demoted to supporting-with-caveat and never quoted as primary. Because
per-probe numbers were produced before this clause, this is recorded as a
§13 deviation-correction, not a silent re-run.

### (h) Korean-forest encoder-combining rule and minimum fold size

- **Encoders are never averaged.** Each of the 18 evaluations is reported
  separately; cross-encoder agreement is its own metric (§18.e Procrustes
  disparity). No post-hoc "keep the encoder that worked".
- **Minimum held-out fold size** (provenance: theory — the metric
  definitions): a fold must satisfy n ≥ 32 (trustworthiness at k = 15
  requires n > 2k + 1) and n ≥ 3 × K_holdout (stable KMeans for the
  adapted ARI). A fold below either floor is reported as **"fold too
  small — metrics undefined"** rather than computed anyway.
- **K on holdout folds is not re-tuned:** K_holdout is fixed once by the
  §9 composite rule on the in-set consensus of that cell.

### (i) Threshold provenance labels

Every numeric bar in §16/§18 carries a provenance tag: **theory** (from a
metric's definition), **pilot** (calibrated on pilot runs), **calibration
fixture** (§6), or **observed-with-headroom**. The 0.70 trust/dist-corr
floors and the K1 0.65 floor are *pilot*-derived; any bar tagged
*observed-with-headroom* (e.g., a "+0.03 improvement" bar quoted against
an observed +0.149) is a **sanity check and cannot back a headline
claim**. The tag rides in the record file with the threshold.

### (j) Null- and negative-result path

If a corpus misses its clause-(a) expectation: the paper still reports all
23 cells with CIs; the OOS-generalization claim for that corpus is
replaced by the pre-written negative finding (clause e wording where
identity is the cause); the threats table gains the corresponding entry;
no strategy, encoder, or metric is dropped post hoc. A null result here is
a publishable finding about projection-based OOS topic assignment, and the
manuscript says so rather than burying it.

### (k) Record-file (bundle) schema additions — §12

`bundle.json` gains required fields: `schema_version`,
`protocol_version`, resolved versions of the number-moving libraries
(`umap-learn`, `scikit-learn`, `numpy`, `torch`, BLAS backend),
`evaluation_mode` (`cv` / `final_fit` / `holdout`), `run_status` +
`failure_reason` (a crashed run leaves a visible marker), and
`metric_recipe_ids` — one registered recipe id per reported value.
**Mechanical gate:** the aggregator refuses record files whose
`protocol_version` predates 1.5, and every result-generating driver
asserts protocol version ≥ 1.5 at startup. The embedding rebuild is
result-blind and may run before ratification.

### (l) §19.a demographic family: m = 10 → m = 9 (deviation note)

§19.a pre-registers Holm over 10 demographic variables; `Country` is
constant (all-USA) in `artist_perspectives.csv`, so the committed analysis
code tests 9. This clause records the overdue §13 deviation: the family is
**m = 9** (the nine variables enumerated in the run matrix), `Country` is
excluded-as-constant, and the §15 changelog entry closes the live
violation.

### (m) Distance-correlation recipes — two names for two calculations

The everyday name "distance correlation" covered two different
calculations; they are now registered recipes (pinned, with reference
implementations and rung tests, by the recipe-pinning change):

- **`distance_correlation`** (reported metric; kind `adapted`):
  *Pearson correlation between condensed pairwise-distance vectors* —
  Euclidean self-distances taken once per unique pair (scipy `pdist`
  condensed form; no diagonal, no double-counting). Explicitly **not**
  Székely's distance correlation. This — and only this — recipe is what
  §18.e's "distance correlation" means, and what the run matrix binds.
- **`dist_corr_loss`** (training-loss component; kind `adapted`):
  1 − Pearson correlation over the *full flattened* self-distance
  matrices (`torch.cdist`; diagonal zeros included, every pair counted
  twice). Calibration-frozen byte-identical
  (`tests/verification/test_dist_corr_loss_regression.py`).

The two demonstrably disagree on identical inputs (closed-form
14/√205 vs 6/√37 on the pinned four-point fixture; 0.854 vs 0.892 was the
corpus-scale observation that surfaced the split). They are never quoted
interchangeably; record files carry the recipe id per value (clause k).

### (n) Generalized Procrustes is the headline Procrustes comparator

Once implemented (roadmap PR 9), **Generalized Procrustes alignment**
(iterative alignment to a shared mean shape) is the headline Procrustes
baseline; pairwise Procrustes remains reported alongside. The §6
pre-registered golden ranges for the *pairwise* variant are untouched —
the new comparator is additive, not a re-calibration.

### (o) Deferred analyses, recorded with reasons

Deferred until after the temporal-holdout program, acceptance criteria
marked deferred (not deleted), each revisitable:

- **Topic-attribution rubric study (§17):** the judging core exists
  (`src/reap/topic_attribution.py`); outstanding are the driver, the
  second judge, the agreement statistic, tests, and the judged runs — a
  genuine medium-sized phase. Stated cost of deferral, honestly: it
  defers the independent semantic validation of the OOS-bridge claim.
- **Real-data OOS-filter re-run (§16.d):** synthetic-shift CI fixture
  stays active; the Korean-forest replication re-run waits.
- **Encoder-sensitivity study (§10, AI-art):** partially superseded — the
  KF arm of §18 already runs three encoders; the AI-art three-encoder
  re-run waits.
- **Two stub dataset loaders** (corporate-sustainability, presidential):
  wait for extraction/labels.

### (p) Residual-risk migration into §11/§18.i

Three risks living only in local notes move into the protocol's
established threat tables, each with a status and mitigation artifact:

- **Set-A-only cross-source verification** (A1/A2/A3 reproduced on seed
  Set A only) — status *open*; mitigation: re-run on Sets B/C when the
  driver lands.
- **A2 Wasserstein-1 on a nominal cluster axis** (W1 treats cluster ids
  as ordered) — status *accepted-limitation*; KL is the primary
  divergence, W1 reported as supplementary with the caveat.
- **Artist probe non-independence** — status *mitigated* by clause (g).

Standing rule: any new caveat surfaced by a run lands as a §11/§18.i
amendment entry **in the same PR as that run**.

---

## Protocol edits carried by this amendment's PR

Surgical, in `manuscript/evaluation_protocol.md`: version header 1.4 → 1.5;
§8's Holm bullet points to the family definitions above; §11 gains the
leakage objection row (clause f) and the three migrated residual-risk rows
with statuses (clause p); §18.g's "216 total" corrected to the per-corpus
counts; §18.h's "3 of 6" corrected per clause (a); §19.a notes the Country
exclusion (clause l); one §15 changelog entry covering all clauses and
naming this document as the authoritative source (mirroring how v1.4 was
ratified from its proposal document).
