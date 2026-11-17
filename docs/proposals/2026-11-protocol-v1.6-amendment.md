# Protocol v1.6 Amendment — Hold-Out Scope and the Per-Seed Unit (Pre-Registration)

**Status:** DRAFT — awaiting ratification. Ratification = merge of this
amendment's pull request by the project lead.
**Applies to:** `manuscript/evaluation_protocol.md` v1.5 → v1.6.
**Hard gate (unchanged from v1.5):** no hold-out (§18) run starts before
this amendment lands. No §18 result exists at the time of writing.
**Machine-readable companion:** `docs/run_matrix_v1_6.json`, checked by
`tests/test_run_matrix_v1_6.py`. The v1.5 matrix (`docs/run_matrix_v1_5.json`)
stays unchanged as the record of what v1.5 registered.

This amendment does four things before any hold-out number exists: it
narrows which of the 23 registered hold-out evaluations are run first, and
says what happens to the rest (clause a); it defines what the "30 per-seed
values" in every paired test are, which v1.5 left unstated (clause b); and it
fixes one acceptance rule that contradicted v1.5 clause (c) (clause c); and it
fixes exactly how each split, reference partition, and K is built, so none of
those can be chosen after results exist (clause e).

---

## Clauses

### (a) Run scope: 11 evaluations first, 12 deferred as a separate family

**In scope (run first):** the 11 evaluations on the canonical encoders —

- AI-art T1–T5 (encoder `intfloat/e5-large-v2`), and
- Korean forest K1 and K2-T1…T5 on the canonical encoder
  `paraphrase-multilingual-MiniLM-L12-v2`.

K1 needs one small build first: the pledge snapshot holds only 18-d
coordinates from an earlier projection head, not raw embeddings, so the 1,662
pledge texts are embedded with the canonical MiniLM encoder by a result-blind
build script before K1 runs (clause e).

**Deferred:** the 12 Korean-forest evaluations on the two non-canonical
encoders (`paraphrase-multilingual-mpnet-base-v2`,
`intfloat/multilingual-e5-large`). Reason: their embeddings have not been
built, and the dissertation chapter does not make an encoder-robustness
claim. The cross-encoder Procrustes disparity metric (§18.e), which needs at
least two encoders per strategy, is deferred with them.

Consequences, fixed now so they cannot be chosen after results exist:

- **Families.** The Korean-forest hold-out family is the 6 in-scope cells:
  6 × 3 baselines × 4 test metrics = 72 tests. The AI-art family is
  unchanged (60). If the deferred cells are run later, their 144 tests form
  their own family, corrected separately and reported as a follow-up; they
  are never merged into the 72-test family after the fact, and no in-scope
  result is re-tested against a larger family.
- **Acceptance count (replaces v1.5 clause a for Korean forest).** At least
  **3 of 6** in-scope Korean-forest evaluations reach trustworthiness mean
  > 0.70 and distance-correlation mean > 0.70 on held-out — the same
  half-of-the-runs rule. AI-art stays at 3 of 5; the K1 trustworthiness
  floor of 0.65 stays.
- **Reporting.** Every deferred cell is listed in the results with
  "deferred — embeddings not built", never silently absent.

```
family_size(ai_art_holdout) = 60
family_size(korean_forest_holdout) = 72
family_size(korean_forest_holdout_deferred_encoders) = 144
family_size(core_methods_per_dataset_per_metric) = 5
family_size(a1_demographics) = 9
```

### (b) What the 30 per-seed values are

v1.5 requires one 30-value column per method per evaluation for the paired
Wilcoxon tests (§8, v1.5 clause d), but the 30 Set D seeds are used to build
a single consensus per evaluation, so the unit behind the 30 values was
undefined. Defined now:

1. **One consensus per evaluation** (per replicate for random splits), built
   from the 30 Set D seeds as UMAP seeds, exactly as §18.d step 3.
2. **Each trained method is trained 30 times** against that one consensus:
   the REAP MLP head, the linear head, and parametric UMAP. Run *i* uses
   Set D seed *i* as its training seed. The UMAP seed stream (consensus) and
   the training seed stream (method) are separate inputs that happen to take
   the same 30 values; the consensus does not change between runs.
3. **Pairing is by seed index.** The paired test for REAP vs a trained
   baseline compares run *i* of REAP with run *i* of the baseline.
4. **The identity baseline is deterministic** (no training), so it has one
   value per metric. Its paired test is the Wilcoxon signed-rank test on
   d_i = (REAP run *i*) − (identity value), i = 1…30.
5. **What these tests mean, stated in the results:** they measure
   run-to-run (training) variation for a fixed split and a fixed consensus.
   They are **not** evidence about generalization to new documents, and the
   chapter does not present them as such.
6. **Generalization uncertainty is reported separately.** For each
   evaluation, metric and baseline: a paired resampling interval over
   held-out documents for the difference REAP − baseline, computed on run 1
   (the first Set D seed) of every method — designated now, so no run is
   chosen after seeing results. B = 2,000 resamples, each an 80% subsample of
   the held-out documents drawn without replacement (the same subsample for
   both methods). Sampling without replacement avoids duplicated points,
   which distort the nearest-neighbour ranks behind trustworthiness and
   silhouette; the 80% size makes the interval somewhat wider than a full-size
   bootstrap, which errs on the conservative side. Where clause (f) of v1.5
   groups chunks by source document, documents (not chunks) are subsampled.
   The 95% percentile interval is reported next to the seed-level result.
   These intervals are descriptive and are not members of any Holm family.

### (c) §18.h second bullet: no cross-method pass on distance correlation

§18.h's second bullet let REAP pass by beating both parametric UMAP and the
linear head "on either trust or dist_corr". v1.5 clause (c) demoted
cross-method distance-correlation comparisons to supporting, because REAP's
heads are trained on a variant of that quantity and parametric UMAP is not.
The bullet now reads:

- Per corpus: at least one in-scope strategy shows REAP significantly
  outperforming parametric UMAP **on trustworthiness**, and the linear head
  on trustworthiness **or** distance correlation (head-vs-head, where both
  train on the same loss), at Holm-corrected α = 0.05.

### (e) How each split, reference partition, and K is built

Fixed now so the driver implements a rule rather than a choice.

**Grouping units (applies v1.5 clause f).** AI-art chunks come from 125 source
documents, identified by `line_number` in the AI-art source chunks file, so
document identifiers are recoverable and T5 is **group-aware**: all chunks of a
document stay on one side. Korean-forest sentences carry no document id, but 10
of the 905 share their normalized text (`section_text_norm`) with another
sentence; sentences with identical normalized text form one group, so an exact
duplicate can never sit on both sides.

**Splits.**

- AI-art T1–T4: by year, exactly as §18.b.
- AI-art T5 (master seeds 20260521 / 20260522 / 20260523): held-out set = the
  first fold of scikit-learn `StratifiedGroupKFold(n_splits=5, shuffle=True,
  random_state=<master seed>)`, stratified by year, grouped by source
  document — about 20% of chunks; the realised count is recorded, not forced
  to 347.
- Korean forest K1: in-set = all 905 reference sentences; held-out = the
  1,662 pledges (clause a).
- K2-T1 / T2 / T3: hold out the Moon (390) / Lee (164) / Park (351) sentences.
- K2-T4 (master seed 20260524): first fold of `StratifiedGroupKFold(n_splits=5,
  shuffle=True, random_state=20260524)`, stratified by president, grouped by
  normalized text.
- K2-T5 (master seed 20260525): `GroupShuffleSplit(n_splits=1, test_size=0.2,
  random_state=20260525)`, grouped by normalized text.

**Leakage audit.** Every cell reports, next to its metrics, the distribution
of each held-out document's highest cosine similarity to any in-set document
and the share above 0.95 — the random splits as v1.5 clause (f) requires, and
the year, president and corpus splits too. Grouping removes exact duplicates
from random splits but not near-duplicates, and the president splits are
defined by president, so one normalized sentence that appears in two
administrations' plans sits on both sides of K2-T1 and of K2-T3; the audit
makes that visible rather than changing the registered split.

**K.** K_holdout is chosen once per cell on the in-set consensus as the
silhouette-maximizing K over 5…30 (KMeans seed 42). This is the §9 composite
rule (0.5 × silhouette + 0.5 × trustworthiness): the consensus embedding's
trustworthiness does not depend on K, so both rules pick the same K.

**Reference partition for the adapted ARI.** The consensus pipeline is rerun
on in-set ∪ held-out with the same 30 Set D seeds and settings; KMeans with
K_holdout (seed 42) partitions it, and the held-out rows' labels are the
reference partition.

**Subsample intervals.** The subsample draws (clause b, item 6) use a local
generator seeded with the first Set D seed; random-split strategies get one
set of intervals per replicate.

### (f) Unchanged

Everything else in v1.5 stands: metrics and their recipes, the replicate
rule, the leakage audit, minimum fold sizes, the identity null
interpretation, the null-result path, the record-file fields, and the
protocol-version gate (drivers assert protocol version ≥ 1.6 once this
lands).

---

## Protocol edits carried by this amendment's PR

In `manuscript/evaluation_protocol.md`: version header 1.5 → 1.6; §8 points
to clause (b) for the per-seed unit; §18.a and §18.g state the in-scope and
deferred counts; §18.h carries the corrected counts and second bullet; one
§15 changelog entry naming this document as the authoritative source.
