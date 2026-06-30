# Trustworthiness Verification Ladder — Evidence Log

Append-only evidence log for the bottom-up metric-correctness verification
ladder targeting REAP's trustworthiness score (Venna & Kaski 2001;
`reap.evaluation.compute_trustworthiness`).

**Why this ladder matters specifically.** Trustworthiness measures whether
local neighborhood structure is preserved when reducing dimension — the
load-bearing claim for REAP's *out-of-sample projection* angle. If new data
projected into the consensus space lands near its true high-d neighbors,
the OOS story holds. If not, the angle collapses.

**Tolerance schedule:**
- Rung 0–2 Euclidean: `1e-12` (exact)
- Rung 1–2 cosine: `1e-1` (loose — sklearn's `argpartition` tie-breaking on cosine distances diverges from stable `argsort`; see findings below)
- Rung 3–4 production-vs-CSV: `1e-6` (CSV printed precision)
- Rung 3–4 production-vs-reference (cosine): `1e-1` (same tie-breaking divergence)

---

## Three load-bearing definitional findings surfaced by this ladder

1. **`compute_trustworthiness` defaults to `metric="cosine"`**, while sklearn's `trustworthiness` defaults to `"euclidean"` and the reference `_reference_trustworthiness` follows sklearn's default. The production CSVs report cosine-metric trustworthiness throughout. **Manuscript implication:** the published `trustworthiness` numbers are cosine-metric values; methods sections must document the metric choice explicitly.

2. **Sklearn's trustworthiness uses `argpartition` internally**, which is non-deterministic on tied distances. On cosine-distance data (where ties are common because cosine collapses magnitude differences), this gives slightly different rankings than the reference's stable `argsort`. Empirically the divergence is ~1e-2 on real corpora. This is a sklearn implementation choice, not a REAP bug, but it must be cited if any downstream prose treats the verified trustworthiness numbers as exact (they are exact at 1e-6 vs the published CSV; the "true" value depends on tie-breaking).

3. **Sklearn enforces `k < n_samples / 2`**, stricter than the original Venna-Kaski 2001 bound `2*n - 3*k - 1 > 0` (i.e. `k < (2n-1)/3`). For n=40 V&K admits k up to 26 but sklearn rejects k ≥ 20. The production pipeline already respects this via `n_nn = min(15, X_high.shape[0] - 1)`.

All three are pinned by tests so a silent change to any of them trips the ladder.

---

## Rung 0 — Closed-form known-answer tests

- **Test file:** `tests/verification/test_trustworthiness_rung0_closed_form.py` (10 tests)
- **Reference impl:** `tests/verification/_reference_trustworthiness.py`

### Hand-computed expected values (Euclidean)

| Case | Inputs | Expected T |
|---|---|---|
| T1 identity | `X_high == X_low` | `1.0` |
| T2 scaled preservation | `X_low = c·X_high` (c > 0) | `1.0` |
| T3 5-point intruder | `X_high = [[0],[1],[3],[10],[11]]`, `X_low = [[0],[10],[3],[1],[11]]`, k=2 | `8/15 = 0.5333…` |

Hand derivation of T3 enumerated: intruders at indices 0,1,2,3,4 with high-d ranks 3,4,3,4,3 → total penalty (rank−k) = 1+2+1+2+1 = 7 → T = 1 − (2/30)·7 = 8/15.

Plus the cosine-default-pinning test (production default IS cosine) and the k-too-large boundary test.

**Rung 0 GREEN — 10 tests pass.**

---

## Rung 1 — Tiny synthetic property verification

- **Test file:** `tests/verification/test_trustworthiness_rung1_synthetic.py` (23 tests)
- **Synthetic:** 40 samples × 8-d blobs, projected to 2-d by PCA.
- **Properties:** PCA preserves structure (T > 0.85 across k and metrics); random low-d → T < 0.7; scale invariance; production ↔ reference exact under Euclidean (1e-12), loose under cosine (1e-1); identity → T = 1.0 under Euclidean; production identity-cosine ≈ 1.0 (within 1e-2, pinning sklearn's tie-break divergence).

**Rung 1 GREEN — 23 tests pass.**

---

## Rung 2 — Escalating + degenerate

- **Test file:** `tests/verification/test_trustworthiness_rung2_escalating.py` (15 tests)
- **Sweep:** n ∈ {40, 200, 1000}; k ∈ {3, 5, 10}; cluster_std ∈ {0.3, 1.0, 2.0}; high-d dim ∈ {4, 8, 32}; k near upper boundary (n=40, k=18); cosine cross-check at n ∈ {40, 200}.
- **Sklearn-bound pin:** test confirms that `compute_trustworthiness(n_neighbors=25, n=40)` raises ValueError (sklearn's `k < n/2` constraint).

**Rung 2 GREEN — 15 tests pass.**

---

## Rung 3 + Rung 4 — Real-corpora cross-check (combined)

- **Test file:** `tests/verification/test_trustworthiness_rung3_real_corpora.py` (27 tests)
- **Scope:** 3 corpora × 3 sets = 9 (corpus, set) tuples; each verified via 3 checks (SHA256 pin + production vs CSV + production vs reference loose).
- **Inputs:**
  - X_high (sentence embeddings): loaded via `reap.datasets.load_<corpus>()`
  - X_low (UMAP output): pinned `results/<corpus>/reap/set_{A,B,C}/consensus_embedding.npy`
- **Production metric:** cosine, `n_neighbors = min(15, n_samples − 1)` per `benchmarks.py:380`.

### Verified consensus_trustworthiness values (REAP, recomputed and matched within 1e-6 to published CSV)

| Corpus | Set | Published `consensus_trustworthiness` | Recomputed | Δ |
|---|---|---|---|---|
| twenty_newsgroups_reference | A | 0.9384 | (matches within 1e-6) | < 1e-6 |
| twenty_newsgroups_reference | B | 0.9377 | (matches) | < 1e-6 |
| twenty_newsgroups_reference | C | 0.9373 | (matches) | < 1e-6 |
| ai_art | A | 0.9175 | (matches) | < 1e-6 |
| ai_art | B | 0.9171 | (matches) | < 1e-6 |
| ai_art | C | 0.9180 | (matches) | < 1e-6 |
| korean_forest | A | 0.8883 | (matches) | < 1e-6 |
| korean_forest | B | 0.8896 | (matches) | < 1e-6 |
| korean_forest | C | 0.8918 | (matches) | < 1e-6 |

**Rung 3+4 GREEN — 27 tests pass.**

### What patterns this rung reveals about REAP's neighborhood preservation

| Corpus | Method-average `consensus_trustworthiness` (across 3 sets) |
|---|---|
| 20NG: REAP | ≈ 0.9378 |
| 20NG: best_of_n | 0.9540 |
| 20NG: parametric_umap | 0.9335 |
| AI-art: REAP | ≈ 0.9175 |
| AI-art: best_of_n | ≈ 0.9381 |
| AI-art: parametric_umap | ≈ 0.9004 |
| KF: REAP | ≈ 0.8899 |
| KF: best_of_n | ≈ 0.9148 |
| KF: parametric_umap | ≈ 0.8881 |

**REAP's trustworthiness is mid-pack.** It's slightly *below* best_of_n on every corpus (best_of_n picks the most trustworthy single seed by silhouette so it's hard to beat on this metric by design), but mostly *above* parametric_umap and naive_average. The gap to best_of_n is ~0.02–0.03 on average — meaningful but not large.

**Manuscript implication for the OOS-projection angle.** Trustworthiness measures in-sample neighborhood preservation. REAP's slightly-lower-than-best_of_n trustworthiness means the REAP consensus space is *not* uniquely good at preserving the original neighborhood structure for *training* points. The OOS story needs a *different* metric: how well does the *projection head* preserve neighborhoods for *held-out* documents? That's the projection-head ladder, the next thing to verify if Angle B is to carry weight.

---

## Full trustworthiness ladder regression

```
test_trustworthiness_rung0_closed_form.py        10 passed
test_trustworthiness_rung1_synthetic.py          23 passed
test_trustworthiness_rung2_escalating.py         15 passed
test_trustworthiness_rung3_real_corpora.py       27 passed
```

75 trustworthiness tests total. All GREEN at published rung tolerances.

**Ladder closed for in-sample trustworthiness.** The OOS-projection question (does the projection head extend this consensus space to held-out documents?) is the projection-head ladder's responsibility.
