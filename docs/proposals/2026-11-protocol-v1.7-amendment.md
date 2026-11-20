# Protocol v1.7 Amendment — Earnings Calls and Presidential Speech (Pre-Registration)

**Status:** DRAFT — awaiting ratification. Ratification = merge of this
amendment's pull request by the project lead.
**Applies to:** `manuscript/evaluation_protocol.md` v1.6 → v1.7.
**Written before any REAP result on this corpus exists.** Runs started before
ratification record this amendment as pending; if it is ratified unchanged
they count, and if any clause changes they are rerun.

The protocol lists two tertiary corpora as "TBD": corporate sustainability
reports and US presidential language (§1). This amendment replaces them with
one tertiary corpus, earnings calls and presidential speech about climate and
energy, and registers how it is built, encoded and evaluated, before any
REAP run on it. Its construction is recorded in the research repository
(`results/corporate_presidential/call_pilot/`, each step with its record
file) and planned in that repository's
`docs/2026-11-19-overnight-run-plan.md`.

---

## Clauses

### (a) The corpus

Passages of two sources, cut by one chunking rule (`cp-chunk-v1`: one to
three paragraphs, at most 250 words, never across a speaker turn) and kept
only when a fixed climate and energy keyword list matches (`cp-keywords-v1`,
unchanged since before any call text was read):

- **Earnings calls** of 100 S&P 500 companies (a nested, salted-hash sample
  of 50 per stratum of climate exposure, explore side of a company-level
  reserve), from three transcript datasets with one version per call event.
  Only management prepared remarks, management answers and analyst
  questions are used.
- **Presidential speech:** the sitting president's own turns in explore-side
  spoken White House documents (remarks, interviews, news conferences),
  Obama to Trump's second term. Reporters', interviewers' and guests' turns
  and editorial notes are left out.

Cleaning (`cp-clean-v1`) removes verified editorial and transcription
material only (stage directions, transcription notes, time stamps, editorial
notes); spoken words are kept. A relevance filter then applies the frozen
relevance definition (codebook version 0) with `gpt-6.1-terra`: passages
judged climate, energy or both are kept. At most 5 passages per document or
call event are kept, in salted-hash order. The REAP sample of N passages
gives presidential passages up to a third and spreads call passages evenly
across companies. N is set from the timing of a smoke test (1,000 and 2,000
passages) before the full runs and is never changed after results exist.

### (b) Two encoders on identical rows

The corpus is encoded twice, on the same rows:

- `intfloat/e5-large-v2` at revision `f169b11e22de13617baa190a028a32f3493550b6`,
  prefix `query: ` — the **reporting anchor**, for continuity with the
  AI-art case study and because pinned weights reproduce;
- OpenAI `text-embedding-3-large` — a **complete parallel replication**. It
  cannot be pinned to a version; its vectors are cached with the SHA-256 of
  the exact inputs.

Both encoders were chosen before any REAP run. Neither is chosen or
dropped on REAP outcomes (silhouette, stability, source separation or
rejection). Small sentence encoders (MiniLM, MPNet) are out of scope.

### (c) What is run

The §3 methods and baselines through `scripts/run_paper_benchmark.py`
(REAP, single seed, best of N, naive coordinate average, Procrustes,
BERTopic), on seed sets A, B and C of the seed manifest (§2), with the §7
metrics that need no labels, the §8 paired statistics and Holm correction,
and K by the §9 rule over 5 to 30 with the K-robustness sweep. UMAP settings
are fixed a priori, not tuned: 5 dimensions, 15 neighbours, minimum distance
0.1. Each encoding is reported on its own; the two are compared afterwards,
descriptively.

### (d) What is not run under this amendment

- External validity against labels (§7): the corpus has no expert labels.
- Topic attribution (§17).
- Placement of later call passages on an earlier map (§18-style temporal
  hold-out) and the presidential-reference conformal filter (§16) on call
  passages. Both need their own split and calibration rules; they are left to
  a later amendment.

### (e) Threats to validity specific to this corpus (added to §11)

- **Sources separate before REAP.** In the exploratory pilot, 99% of a
  passage's 10 nearest neighbours came from its own source under both
  encoders. Stable REAP structure on this corpus can reproduce that split
  through genre, speaker mix and selection; it is not evidence that the
  sources discuss different issues.
- **Company concentration.** Call passages' neighbours share the company far
  above chance (67% to 80% against 13% in the pilot).
- **Provider change.** After mid-May 2025 only one transcript provider
  covers calls, so time and provider are confounded there.
- **Relevance filter error.** In its pilot the two models agreed on relevant
  against not relevant for 90.7% of 300 passages (84% on one call dataset).
- **Scope.** Results describe the retained passages of these collections, not
  how often companies or presidents discuss climate and energy, and not the
  S&P 500 as a whole.

### (f) Citability

Results become citable only after this amendment is ratified and the
provenance audit has run on each result tree. Until then every result tree
of this corpus is marked exploratory.
