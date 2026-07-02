# Pending prose corrections (apply at manuscript unfreeze)

The manuscript prose is frozen until trusted results land; protocol
amendments are the only exception. This file is the durable ledger of
known-wrong or under-specified numbers in frozen or shipped text, so nothing
is silently dropped when the freeze lifts. Every entry names the claim and
its locations, the correct value, and the committed artifact that backs it.

Rules: entries are appended, never deleted — when a correction lands, mark
it applied with the commit that did it. Non-frozen occurrences (README, code
docstrings) are fixed as soon as they are found and noted here for the
record.

---

## 1. Korean-forest ARI headline — "0.75, +32% over Procrustes 0.56"

**The claim:** REAP reaches ARI 0.75 against expert labels vs. 0.56 for
Procrustes consensus, a "+32%" improvement.

**Locations:** `manuscript/sections/introduction.md:40` (the 0.56 claim) and
`:51-52` (the 0.75 / "32%" claim) — frozen, apply at unfreeze;
`manuscript/key_validated_results.md:34` — frozen table, apply at unfreeze.
Fixed already (same change that created this ledger): `README.md`, which had
additionally misattributed the number to the AI-art corpus, and the
docstrings at `src/reap/consensus.py` (module header and
`get_procrustes_consensus`).

**The correct statement:** the repo's independently re-derived Korean-forest
number is **ARI 0.122634**, consensus-vs-expert-labels, with 23 consensus
clusters against 20 expert classes. The 0.75/0.758 figure comes from the
sibling analysis project under a definitionally different calculation
(different label set, K, and subset) — not a bug, but it may only be quoted
with that exact provenance stated. Arithmetic nit if the pair is ever quoted
under one definition: 0.75 vs 0.56 is +33.9%, not +32%.

**Source:** `docs/verification/ari_ladder_log.md`, "Adjudication" section
(independent re-derivation from `load_korean_forest()` ground truth +
`consensus_labels.npy`, agreement to 1e-7).

## 2. Lovato artist transparency consensus — never "~85%"

**The claim to guard against:** transparency as the highest-consensus theme
"at ~85%". No tracked file currently quotes ~85% (checked 2026-07-02); this
entry exists so the wrong value never enters prose at unfreeze.

**The correct statement:** **80.95%** (204 of 252 artists in
`artist_perspectives.csv`, in the AI-art source-data project), or 80.0%
(368/460) on the raw survey rows. 89.87% is correct only if "unsure"
responses are excluded and must be labeled as such. The
highest-consensus-theme claim itself is firm.

## 3. Cross-source demographics: Holm family is m = 9, not the
pre-registered m = 10

**The claim:** protocol §19.a pre-registers ten demographic tests (Holm
m=10) for the A1 stratification analysis.

**The correct statement:** the committed analysis runs nine — Country is
excluded as degenerate (all-USA). See
`scripts/run_cross_source_analyses.py` (demographics list). The protocol
fix is a §19.a deviation note + §15 changelog entry, scheduled inside the
protocol v1.5 amendment (roadmap PR 2) — a live deviation until then.

## 4. AI-art reference corpus size — N = 1,736, not 1,742

**The claim:** N = 1,742 appears in stale prose
(`docs/verification/ari_ladder_log.md:374` and the module docstring of
`tests/verification/test_ari_rung4_real_corpora.py`).

**The correct statement:** the production arrays are shape (1736,) and the
loader fails closed on any other count (`src/reap/datasets/ai_art.py`).
Fix the two stale mentions at unfreeze (or whenever those files are next
touched); no pre-registered count is wrong — protocol §18.b already uses
1,736.

## 5. Seed-manifest `high_exclusive` off-by-one — and its generator

**The claim:** `manuscript/seeds/seed_manifest.json:6` records
`"high_exclusive": 2147483648`.

**The correct statement:** the sampling bound actually used is
`SEED_RANGE_HIGH = 2**31 - 1 = 2147483647` (already exclusive; see
`manuscript/seeds/generate_seed_sets.py:43`). The metadata edit must be
paired with fixing `generate_seed_sets.py:144`, which writes
`SEED_RANGE_HIGH + 1` back into the manifest — otherwise regeneration
reintroduces the off-by-one. Seed values themselves are unaffected
(byte-identical; checkable with a plain diff). Scheduled with the
data-integrity fixes after the experiment phases.

## 6. A2 cross-source metadata — record the smoothing and distance
conventions

**The claim:** the A2 artist-vs-public alignment artifact reports KL and
Wasserstein numbers without recording the conventions used.

**The correct statement:** record `laplace_add_one` smoothing and the
Wasserstein-1 normalization in the A2 JSON metadata when the artifact is
re-emitted under the versioned record-file writer (roadmap PR 7). The
artifact now lives in the REAP-research sibling
(`results/projection_head/ai_art/oos_demo/cross_source/`).
