# Protocol v1.8 Amendment — REAP's Final Step and Repeatability Across Seed Sets (Pre-Registration)

**Status:** DRAFT — awaiting ratification. Ratification = merge of this
amendment's pull request by the project lead.
**Applies to:** `manuscript/evaluation_protocol.md` v1.7 → v1.8. The v1.7
draft (earnings calls and presidential speech) is pending in its own pull
request; if it is not ratified first, this amendment takes the next free
version number and its clauses are unchanged.
**Hard gates:** REAP's default final step does not change under this
amendment (clause j). No candidate result exists at the time of writing.
Full-size candidate runs wait for the project lead's go-ahead after the
pilot (clause g).

This amendment is written after a diagnosis that changed no package code and
tested no fix. It records what the diagnosis found, registers how
repeatability across seed sets is reported from now on (both K rules, and the
spread over the final step's seed), and fixes, before any result, the
candidate final steps, the corpora and seed sets they are run on, the
criteria a candidate must meet to replace REAP's current final step, and the
rule that picks one.

---

## Background

**What prompted it.** REAP's consensus clusters repeat less across seed sets
than the baselines' on two well-separated corpora, and more on the two
others (§7 Level D, mean ARI over the set pairs A–B, A–C, B–C; source:
`between_set_reliability.csv` per corpus in the research repository):

- 20 Newsgroups reference: REAP's three pairs 0.855 to 0.963 (mean 0.903),
  against 0.980 to 0.997 for best of N, naive average and Procrustes (best
  mean 0.994, Procrustes).
- Calls and presidential speech: REAP mean 0.823 against baseline means of
  0.828 to 0.965 with e5-large-v2, and 0.633 against 0.827 to 0.994 with
  text-embedding-3-large.
- AI-art: REAP's pairs 0.880 to 0.975 (mean 0.915); the best baseline mean
  is 0.859 (best of N), and single baseline pairs reach 0.864 (best of N) and
  0.881 (BERTopic).
- Korean forest: REAP's pairs 0.766 to 0.842 (mean 0.810), against at most
  0.781 for any baseline pair (best mean 0.740, best of N).

REAP's averaged distance matrices differ by only 0.5% to 2.1% (relative
Frobenius distance) between seed sets, so the variation enters after
averaging: in the one final UMAP map of the averaged matrix (seed 42), the K
rule, and k-means (one k-means++ start, `n_init="auto"`).

**Diagnosis.** `scripts/experiments/run_final_step_diagnosis.py` holds each
seed set's averaged matrix fixed and reruns only the final step under ten
final seeds (42 and 0 to 8), in three arms: all of the final step reseeded,
only the map reseeded, only the K search and k-means reseeded. Every run
whose seeds are all 42 reproduced the benchmark's saved consensus labels
exactly, on all four corpora and all three sets. Results (research
repository, `results/final_step_repeatability/diagnosis/`,
`diagnosis_summary.csv` at full precision):

1. On every corpus, rerunning only the final step on one fixed matrix gives
   agreement about as low as the agreement between seed sets. Runs on two
   different sets' matrices with different final seeds agree no less than
   two runs on the same matrix. The small differences between the averaged
   matrices contribute essentially nothing; the final step produces the
   spread.
2. The map, the K rule and the single k-means start all contribute. The K
   chosen on one fixed matrix moves widely on Korean forest (9 to 30) and on
   the OpenAI encoding of the calls corpus (6 to 30); at a fixed K the
   agreement rises there, but map and k-means noise remain.
3. REAP's lead on AI-art and Korean forest depends partly on the final seed.
   At seed 42 its between-set ARI is 0.915 (AI-art) and 0.810 (Korean
   forest); averaged over the other final seeds it is 0.853 to 0.862 and
   0.724 to 0.730, against 0.859 and 0.740 for best of N.

**K rule.** §9 says K is chosen once, on REAP's Set A map, and frozen for
every method. The benchmark driver instead chooses K by silhouette
separately for each method and seed set, and every between-set number so far
was produced that way. Because trustworthiness does not depend on the
labels, §9's score (0.5 × silhouette + 0.5 × trustworthiness) is maximised
over K by silhouette alone on one map; the substantive difference is the
freezing. With K frozen at REAP's Set A K, and each method's saved map
clustered with the harness's own k-means
(`scripts/experiments/run_protocol_k_check.py`,
`results/final_step_repeatability/diagnosis/protocol_k_check.csv`), REAP's
mean between-set ARI against the best baseline is 0.959 against 0.994 on 20
Newsgroups, 0.909 against 0.824 on AI-art, 0.829 against 0.718 on Korean
forest, 0.760 against 0.802 on calls (e5) and 0.727 against 0.757 on calls
(OpenAI). That check used final seed 42 only; its repetition over the ten
final seeds is part of this amendment's runs (clause b).

**UMAP settings.** 20 Newsgroups and the calls corpus ran on default UMAP
settings (15 neighbours, minimum distance 0.1, 5 dimensions); AI-art and
Korean forest ran on settings from REAP's parameter search, which was never
run on the other two. Well-separated corpora and default settings coincide
on all four corpora, so the cross-corpus pattern alone cannot tell them
apart. Clause (d) registers a small settings diagnosis for that.

This is a deviation under §13 (the benchmark's K rule differs from §9's);
this amendment is its record.

---

## Clauses

### (a) Two K rules, both reported

Every between-set table reports both:

- **Rule S (per-set silhouette K):** each method picks K by silhouette over
  the dataset's K range, separately for each seed set, as the benchmark
  driver runs it.
- **Rule F (frozen K, §9):** K is REAP's Set A choice under rule S (the
  silhouette maximum on REAP's Set A map), applied to every method and seed
  set.

Neither rule alone is the headline. Rule F's K comes from REAP's own map,
which favours REAP; the fixed-K sweep (K = 5, 7, 10, 15, 20; k-means with
`random_state=0` and ten starts, as in the existing fixed-K check) is
reported next to both as the neutral comparison.

### (b) The final seed is varied, not fixed

REAP's between-set agreement is reported over the ten final seeds 42, 0, 1,
…, 8: for final seed s, the final map, the K search and k-means of every set
use seed s, and the three set pairs are compared at the same s. The
baselines' k-means and K search use the same seed s on their own consensus
maps. Each method's figure is the mean over the ten seeds of the mean over
the three pairs, with the minimum and maximum over seeds. Seed 42 alone is
still shown, as the benchmark's value.

### (c) Candidate final steps

Registered here, implemented in `scripts/experiments/final_step_candidates.py`
(research code, not used by the package). All take REAP's averaged matrix D,
the dataset's K range, and ties between K values go to the smaller K. The
display map, on which consensus silhouette and trustworthiness are measured,
is REAP's seed-42 final map for every candidate.

- **a1 — average linkage on D.** Average-linkage tree on D, cut by merge
  order into exactly K clusters. Rule S: K maximises silhouette computed on
  D. Deterministic.
- **a2 — k-medoids on D.** Greedy start (the first medoid minimises the
  total distance to all points; each next one lowers that total most; ties
  to the lowest index), then alternating updates until the medoids stop
  changing (at most 300 rounds, else the run fails). Rule S: K maximises
  silhouette on D. Deterministic.
- **b — co-association over ten final maps.** For each final seed s (42, 0
  to 8), the final map with seed s gets the current K rule and k-means with
  seed s. The share of maps that put two documents in one cluster, subtracted
  from one, is a distance; an average-linkage tree on it is cut by merge
  order. Rule S: at the median of the ten K values (the lower middle value).
- **c — K by agreement.** For each K in the range, k-means at K on each of
  the ten final maps (seed s on map s); K maximises the mean pairwise ARI of
  those ten labelings; the labels are the seed-42 map's k-means at that K.
  Defined under rule S only.
- **d — ten k-means starts.** The current final step with `n_init=10` in the
  K search and the final fit. The clustering harness is shared, so d is
  applied to every method, and its comparison is against baselines
  re-clustered the same way. d was added after the diagnosis showed that a
  single k-means start alone moves the labels at a fixed K; on its own it is
  not expected to close the gap, since the existing fixed-K check already
  used ten starts.

Under rule F, every candidate clusters at the frozen K: a1 and b are cut at
it, a2 runs at it, d fits at it, and c does not apply.

How the final seed s of clause (b) enters each candidate: a1, a2 and b have
no seed of their own (b already pools all ten maps), so their labels are the
same for every s; c chooses K from all ten maps and takes its labels from
map s with k-means seed s; d uses map s with k-means seed s. REAP's current
final step at seed s is map s with the K search and k-means at seed s.

### (d) UMAP-settings diagnosis (no setting is adopted)

`scripts/experiments/run_umap_settings_diagnosis.py` runs the whole pipeline
(the 30 per-seed maps and REAP's final map) for each cell of a grid fixed
here: n_neighbors in {15, 30, 50} × min_dist in {0.0, 0.1}, with the
dataset's benchmark dimensions and K range, on the 20 Newsgroups reference
and the calls corpus smoke sample of 1,000 passages (e5), seed sets A to C,
for REAP and the three baselines that share its maps (best of N, naive
average, Procrustes). It reports between-set ARI under both K rules, the
fixed-K sweep, consensus silhouette, trustworthiness, ARI with the expert
labels (20 Newsgroups only) and REAP's within-matrix agreement. It is a
diagnosis: no setting is chosen from it. Choosing settings for a corpus is
done by REAP's registered parameter search, run in the research repository;
if that changes a corpus's settings, the change and its reruns get their own
decision.

### (e) What a candidate must meet

REAP's claims are about messy data with overlapping groups and about placing
several datasets in one shared space; it is not claimed to beat the
baselines on clean, well-separated corpora. The corpora are therefore split:
AI-art and Korean forest are the **gating** corpora; the 20 Newsgroups
reference and the calls corpus (e5 and OpenAI) are **no-harm** corpora,
where the baselines already repeat almost perfectly. On every corpus cell,
seed sets A, B and C with 30 seeds each, under **both** K rules:

1. **Repeatability.** On the gating corpora, its between-set ARI (clause b)
   is at least the highest between-set ARI among the baselines treated the
   same way (same K rule, same final-seed variation, and for candidate d the
   same ten-start clustering): best of N, naive average, Procrustes, BERTopic
   and parametric UMAP. On the no-harm corpora, it is at least REAP's current
   final step under the same K rule and final-seed variation, and its gap to
   the best baseline is reported.
2. **Cluster separation.** Its consensus silhouette (Cell D: its labels on
   the display map, euclidean) is no more than 0.05 below REAP's current
   final step on the same corpus, set and K rule, in every set.
3. **Faithfulness.** Trustworthiness of its display map is no more than
   0.01 below REAP's current final step. Trustworthiness depends on the map,
   not on the labels; every candidate above keeps REAP's seed-42 map, so this
   criterion cannot fail for them. It is kept so that any later candidate
   that changes the map is held to it.

The margins 0.05 and 0.01 are judgment values fixed before any candidate
result, not derived from data; the project lead confirms or changes them at
ratification.

Reported but not gating: silhouette computed on D; AMI between sets; the
fixed-K sweep; ARI with the expert labels on 20 Newsgroups; wall time and
peak memory at each corpus's full size.

**Open for ratification.** (i) Naive coordinate averaging is the §3
strawman; keep it among the baselines for criterion 1 on the gating corpora,
or report it without gating? (ii) Must criterion 1 hold under both K rules
(as written), or under one named rule? (iii) Is the gating / no-harm split of
the corpora as stated?

### (f) Which candidate, if any

A candidate passes if it meets all three criteria on every corpus cell
under both K rules. Among passing candidates, the one proposed as REAP's
final step has the largest smallest margin over the best baseline on the
gating corpora (criterion 1, minimum across those cells and K rules); a tie
goes to the one with the lower full-size wall time. If none passes, REAP's final step stays as it is,
the result is reported as a negative finding with its numbers, and the
threats it exposes are added to §11. Every candidate's results are reported
whether or not it passes.

### (g) Pilot first, then the full run

The pilot (`scripts/experiments/run_final_step_pilot.py`) runs every
candidate at smoke size: the 20 Newsgroups reference (800 documents, its
full size) and the calls corpus smoke sample of 1,000 passages under both
encodings, seed sets A to C with 30 seeds, through the benchmark's harness
with REAP and the three baselines that share its maps (BERTopic and
parametric UMAP are not rerun at smoke size). Its purpose is feasibility:
wall time and peak memory, measured, and any failure. A candidate is dropped
before the full run only if it fails to run or is infeasible at the largest
corpus (5,000 passages), judged from the measured times and memory and
stated with them. No candidate is dropped on the pilot's repeatability
numbers, which are reported as a first look only. The full run starts only
after the project lead's go-ahead.

The full run reuses each set's averaged matrix (the benchmark's saved copy,
or the diagnosis's rebuilt copy, which reproduced the saved labels exactly)
and each baseline's saved consensus map and labels; no per-seed UMAP map is
recomputed for it.

### (h) Statistics

Each corpus cell has three seed-set pairs, too few for a test across seed
sets; no p-value is computed for between-set agreement. Each candidate's
margin over the best baseline is reported as a point estimate with a 95%
percentile interval from a paired bootstrap over documents (1,000
resamples, fixed generator seed 20261120): the same resampled documents
are used for the candidate and the baseline in all three pairs. The interval
describes sampling of documents, not of seed sets. Five candidates are
compared on five corpus cells under two K rules; the pass rule and the
selection rule are fixed above so that no comparison is chosen after the
fact, and all of them are reported.

### (i) Threats to validity (added to §11)

- **Same corpora for diagnosis, pilot and confirmation.** The candidates
  were designed after seeing the diagnosis on these corpora. Fixing the
  candidates, criteria and selection rule before any candidate result, and
  reporting every candidate, limits but does not remove the optimism. Seed
  set D stays reserved for §18.
- **Rule F favours REAP.** Its K is chosen on REAP's map. The fixed-K sweep
  is the neutral check, and rule S is reported alongside.
- **Choosing K by agreement favours small K.** Merging well-separated
  groups is itself repeatable, so candidate c can prefer a K below the true
  one (pinned by a test). The K range's lower bound of 5 and criterion 2
  limit this; c's chosen K is reported per set.
- **Silhouette on the display map favours k-means labels.** Candidates whose
  labels are not built on that map (a1, a2, b) can lose on criterion 2 for
  that reason alone; silhouette on D is reported next to it.
- **Repeatable is not the same as right.** Coarser clusters repeat more
  easily. Criterion 2 guards against collapse, and on 20 Newsgroups the ARI
  with the expert labels is reported.
- **Fairness of the comparison.** Candidates a1, a2, b and c change REAP's
  clustering only; the baselines keep k-means. A REAP gain under c can come
  from its K rule rather than its averaged matrix. Candidate d changes the
  shared harness and is compared against baselines clustered the same way.
- **Ten final seeds.** The spread over the final step rests on 45 pairs of
  ten runs per matrix.
- **One platform.** All runs are on macOS arm64; numbers move with library
  versions and the linear-algebra library, which every record file states.

### (j) What ratification does and does not do

Ratifying this amendment registers clauses (a) to (i). It does not change
REAP's default final step. If a candidate passes, adding it to the package
(as an option, with tests) is a separate pull request, and making it the
default is a separate decision with its own changelog entry and reruns of
every affected result.

### (k) Unchanged

Everything else in the protocol stands: the per-seed maps and their
settings, the averaging, the methods and their own clustering, the metric
recipes, the seed manifest, and every §16 to §18 rule.

---

## Protocol edits carried by this amendment's pull request at ratification

In `manuscript/evaluation_protocol.md`: version header → 1.8; §7 Level D
points to clauses (a) and (b); §9 records that the benchmark ran rule S
and that both rules are reported; §11 takes clause (i); one §15 changelog
entry naming this document as the authoritative source.
