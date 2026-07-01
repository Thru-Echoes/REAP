# REAP dev-plan revision (2026-07)

**Short version: the existing plan is good. This document strengthens it in a few specific places.
It is not a rewrite.**

This document reviews the current development roadmap (Phases 0–7 in `HANDOFF_NEXT_SESSION.md`) and
adds a set of targeted fixes. The review was done by reading the plan closely from many angles —
several "what could be better?" passes and several "try to break this" passes, using two different AI
model families — and then **checking every proposed fix against the actual code and files** before
keeping it. Fixes that turned out to be wrong, or that the plan already handled, were dropped; they
are listed at the end so the review is honest in both directions.

The bottom line: the plan's overall shape is right — the order of the phases, what depends on what,
and the decision to keep the package unpublished until the results are solid. Nothing found here is a
"stop the presses" problem on its own, because the plan already has good safety nets (the manuscript
text is frozen until results are trusted; the metric-checking "ladders" already exist; the release
path is disabled on purpose). What is left are small, specific gaps. Most are ways to make an existing
plan item a little stronger; a few are new items; a couple are about doing things in a better order.

---

## Context: REAP is already being used in two papers

This matters for how we prioritise. REAP is not a method waiting for its first result — it is already
carrying real analyses:

- The **AI-art manuscript** is under editorial review at a Nature collection (several weeks in).
- The **Korean-language / Korean-forest manuscript** (Hye In's) is close to submission.

So the separate **REAP methods paper + software package** does not need to prove REAP works from
scratch. Its job is to explain the method clearly, make the results reproducible, and show the one
thing REAP is really for: projecting new data into a stable, shared "map" of topics — which is
especially useful for messy, long-time-span, or multi-source text.

One consequence is time-sensitive: the Korean-forest ARI number that appears in the manuscripts (see
"Theme 3") should have its exact definition confirmed **before Hye In submits**. This is not because
the number is wrong — it is because the word "ARI" covers a few slightly different calculations, and
we want to be sure the reported number uses the intended one.

---

## A note on how we compare REAP to other methods (important)

An earlier draft of this revision leaned too hard on strict "REAP must beat method X" pass/fail rules.
That is not the goal. The plan is:

- **Run all the method comparisons we have mapped out** (identity, linear, parametric UMAP, Procrustes,
  and so on). Keep them comprehensive and fair.
- **Report them honestly** — with effect sizes (how big the difference is) and confidence intervals
  (how sure we are), and say plainly when REAP ties or trails a method.
- **Feature only the most informative comparisons in the paper.** We are not tying the paper's success,
  or REAP's value, to beating a specific method on a specific test.

REAP's contribution is the framework itself — rotation-invariant consensus and out-of-sample
projection into a stable reference space — plus its reproducibility and stability. A methods paper
built on that does not live or die by a single leaderboard number.

This framing also happens to fix a real weakness the review found. A rigid "pass if REAP beats
baseline X" rule is easy to game (you can pick the comparison that happens to win) and can even look
reverse-engineered. Reporting **everything** transparently is both more honest and more convincing
than a strict gate. (Counterpoint, so it is on the record: reviewers of a methods paper will still ask
"why use this instead of the simpler option?", so we should make sure at least some comparisons show
REAP is clearly competitive or better on something that matters — we just should not pre-commit the
paper's fate to one test.)

---

## What is already right (no need to revisit)

The review confirmed these as correct and deliberate; several criticisms were checked and turned out
to be wrong:

- **The real critical path is producing results, not publishing.** The release path stays off
  (a "do not upload" marker plus a disabled release workflow) until results are trustworthy.
  Front-loading the cheap CI-cleanup items is fine — they change no behaviour and unblock honest
  result generation.
- **The phase order is real.** Fix the git branch setup first; build the shared pieces before the
  experiment scripts that use them; the AI-art experiments (whose embeddings are already cached) can
  run before the Korean-forest ones (which need new embeddings). Splitting the code into a separate
  research repo in two stages — prepare now, do the destructive part later — is the right call.
- **The experiment scripts already build their file paths relative to the repo root**, so the later
  repo split will not force path rewrites in them.
- **The macOS test skip is done correctly** — the exact-math checks run on every operating system;
  only the platform-sensitive number ranges are pinned to the Linux reference machine. No agreed
  target was loosened.
- **The out-of-sample Korean-forest corpus already has a working loader** — it is not an undefined
  input.
- **Exporting the projection head and trimming the public name list are already in the plan.**
- **The main statistical test compares 30 random seeds, not 3 dataset splits**, and the "best of N"
  baseline is chosen by cluster quality (silhouette), not by the answer key — so two "this is
  leaky / underpowered" objections do not apply.
- **The roadmap is ordered by dependencies, not by calendar date** — it commits to no fixed
  submission date, so there is no missed deadline to reset.

---

## The fixes, grouped by theme

Each fix says what problem it prevents, the concrete change, where it fits in the existing plan, and a
rough size (S/M/L = small/medium/large). Where a fix mentions a specific file or line, **check the
location before acting** — the review found it by reading the repo, but line numbers move.

### Theme 1 — Lock down the experiment rules before running them (do this first)

These are small updates to the pre-registered plan (protocol section 18) — the written promise of
exactly how the temporal-holdout experiments will be scored. Pre-registering means writing the rules
down *before* seeing results, so we cannot (even accidentally) tune the rules to flatter the outcome.
The plan already commits to this; a few details are under-specified and should be settled and
committed as a small amendment (call it **v1.5**), with a TRACE note recording the change, **before any
Phase 3/4/5 experiment runs**. Oliver has approved this amendment in principle.

The gaps to close (keeping the "report, don't gate" framing above):

- **Say exactly which comparisons and metrics are the main ones, and which are secondary.** Right now
  the scoring rule ("at least 3 of 6 strategies clear the bar, and at least one strategy significantly
  beats two baselines") mixes several things and leaves room to pick a winner after the fact. Fix:
  write down the full list of runs up front (each dataset × encoder × seed-set × holdout strategy ×
  baseline × metric), mark which are primary evidence and which are supporting, and report all of them.
  This removes the "pick the comparison that wins" loophole without turning the paper into a pass/fail
  test. *(S — but do it before experiments)*
- **Fix the "3 of 6" mismatch.** The AI-art plan defines five holdout strategies; the Korean-forest
  plan schedules eighteen (six strategies × three encoders). "6" matches neither. Fix: state the count
  clearly for each dataset. *(S)*
- **Decide up front how to combine the three Korean-forest encoders** (report each separately? average?
  take the median?). Deciding after seeing results lets you quietly keep the encoder that worked. Fix:
  write the rule down first. *(S)*
- **Include the simplest baseline in the reporting, not just in the runs.** The plan builds an
  "identity" baseline — literally copy the answer from the nearest known point — as a sanity floor, but
  the current scoring never mentions it. It is worth reporting how REAP compares to this simplest
  option, because a method that only ties the "just copy the neighbour" trick has not shown much. Fix:
  report the REAP-vs-identity difference (with effect size) as standard, without making it a hard gate.
  *(S)*
- **Say where each numeric threshold came from.** Some bars (for example a "+0.03" minimum) sit against
  an observed value (say +0.149), which can look like the bar was drawn around the answer. Fix: for
  each threshold, note whether it comes from theory, a pilot, a calibration fixture, or an observed
  value with headroom — and if it is the last kind, label it as a sanity check rather than proof. *(S)*
- **Plan for a null or negative result.** The current write-up assumes the bars are met. A result where
  REAP does *not* clear a bar is still a real, publishable finding. Fix: write down, in advance, what
  we report and claim in that case. *(S)*
- **Set a minimum fold size for the small Korean-forest corpus.** Holding out one president from 905
  documents can leave a fold too small for a stable number of clusters. Fix: pre-register a minimum
  fold size and how the cluster count is chosen (once, fixed — section 9 already pins this for the main
  runs; extend it to the holdout folds). *(S)*

### Theme 2 — Make each comparison a fair one

We are keeping all the comparisons; this is about making the ones we run honest and strong, so nobody
can say we compared against a weak version of a method.

- **Confirm the Procrustes baseline is the strong ("generalized") version.** REAP's story includes
  "plain Procrustes alignment is not enough," so the Procrustes we compare against should be the
  standard strong variant (Generalized Procrustes: line everything up to a shared average shape,
  repeatedly, until it settles), not a quick pairwise alignment. If the current one is the quick
  version, the comparison is unfair to Procrustes and a reviewer will notice. Fix: implement
  Generalized Procrustes in `consensus.py`, use it as the Procrustes baseline, and add a test that its
  result does not depend on the order the seeds are fed in. *(new item, L — worth it for a fair
  headline comparison)*
- **Add two more comparison points if we want them.** Projecting through the Procrustes-aligned space,
  and a version of REAP that uses a single seed (to isolate what the multi-seed averaging actually
  buys), are both natural questions a reader will have. Optional, but cheap to add to the comparison
  set. *(M, optional)*
- **Handle the "same artist appears many times" issue in the AI-art analysis.** 1,259 probes come from
  only 252 artists, so treating each probe as independent overstates the statistics (roughly 5×). This
  is already flagged as a caveat, but it is better to fix it: aggregate to the artist level (or use a
  model that accounts for repeated artists), and carry that through to the related temporal analysis
  and any bootstrap that resamples probes. *(M)*

### Theme 3 — The Korean-forest headline number

The manuscripts report REAP's Korean-forest agreement with the expert labels as ARI ≈ 0.75 (about
+32% over Procrustes's 0.56). The repo's own checking script computes ≈ 0.12 for what looks like the
same thing. That is a big gap (~6×) on an important number. The good news, confirmed by the review:
**the plan already knows about this, and the checking "ladder" already worked out why** — the 0.75 and
the 0.12 are two *different calculations* (different label set, cluster count, and subset), not a bug.
So this is not a hunt for an error; it is finishing the paperwork so the right number is used and
labelled clearly. Given Hye In is submitting soon, do the first item now.

- **Write the correction into both "pending corrections" lists.** The plan keeps two lists of
  manuscript numbers to fix at unfreeze, but neither includes the most important one. Add, at the top
  of both: the claim ("REAP Korean-forest ARI 0.75, +32% over Procrustes 0.56"), the value the repo's
  independent check produces and the exact definition it uses, and the source (the checking ladder's
  top rung). Then confirm the manuscripts use the intended definition. *(strengthen existing, M —
  time-sensitive for Hye In)*
- **Turn the open question into a decision.** The plan lists this as an unresolved "does the +32%
  trace to a real file?" — but the ladder already resolved it. The real decision is: restate the
  number using the confirmed definition, and make sure "+32% vs Procrustes" compares the two under the
  *same* definition. *(part of the above)*
- **Pin the one canonical Korean-forest ARI definition before new numbers are written.** The checking
  log notes two definitions currently share one column name. Fix that naming and add a small test that
  locks the definition, so the new Phase 3/5 scripts do not inherit the confusion. *(strengthen, S)*
- **Add an automatic check that reported numbers trace to a real file.** The plan says "a result
  without its record file is not citable," but nothing enforces it, and the one automatic check only
  looks at synthetic test fixtures. Fix: add a check that every important number in the README and the
  manuscript tables points to a committed data file. This is the check that would have caught the
  headline number drifting. *(strengthen, M — also a coding convention)*

### Theme 4 — Make the "record file" a real, checked format

Every result carries a small `bundle.json` "record file" that says how it was produced. This is the
backbone of reproducibility, but right now it is an informal dictionary.

- **Give `bundle.json` a defined, version-numbered format that is checked automatically.** It is
  written by several scripts and read by one combining script, so a missing or misspelled field breaks
  quietly. Fix: define it as a strict data model (frozen Pydantic) with a version number and a
  description on every field — the commit ID, the script, the dataset and its checksum, the seed list,
  the settings, the Python/platform, and the exact versions of the number-moving libraries (UMAP,
  scikit-learn, NumPy, and the math backend). Write it safely (to a temp file, then rename). Add a test
  that reads it back and fails if a required field is missing. *(strengthen existing, M)*
- **Record the *scientific* environment, not just a package list.** A plain `pip freeze` is not enough
  — the exact library and math-backend builds move the numbers. Fix: capture a conda environment
  export (or the key library + backend versions) into every record file, and commit an
  `environment.yml` before the experiments, not at the very end. Keep this separate from the
  lighter developer/CI lock file. *(medium, S)*
- **Make the shared embeddings rebuildable on a clean machine.** The Korean-forest experiments depend
  on encoder outputs that live in a local cache, outside the repo. Fix: make the re-embedding
  repeatable (pin the encoder version, the seed, the text normalisation, and the `query:` prefix for
  the multilingual-E5 model), record where the source text came from (checksum + encoder version +
  source-repo commit + build date), and document one command to rebuild the cache from source. *(medium, M)*
- **When combining results, stop on missing pieces instead of quietly averaging what is there.** If the
  combining script silently averages over whatever runs happen to be on disk, it can drop failed runs
  and flatter the average. Fix: have it work out the full expected list of runs from the plan + the
  seed list, and **stop with a clear error** if any run is missing or extra. (This is the opposite of
  the sibling TRACE project's "warn and keep going" style, which is right for an audit log but wrong
  for a results table.) *(medium, M)*
- **Record whether a number is in-sample or out-of-sample.** An in-sample number reported as
  out-of-sample would be a serious (if honest) mistake. Fix: add a required "evaluation mode" field to
  the record file and check it when combining. *(medium, S)*
- **Add one end-to-end "rebuild from a clean checkout" test** for a small but important number.
  *(medium, M)*

### Theme 5 — Test the scripts that produce the paper's numbers

The experiment and combining scripts are how a wrong number would actually reach the paper, yet they
have no tests.

- **Add a quick test to each experiment script and the combining script.** On a tiny synthetic
  example, check the basics: the held-out and kept-back items do not overlap and cover everything; the
  comparison file contains all the planned baselines and metrics; the record file is valid; and the
  scoring flags come out right on an example with a known answer. *(strengthen, M)*
- **Treat these scripts as real, tested code.** They are the path by which a wrong number would reach
  the paper, so they get the same test care as the package itself. *(convention)*

### Theme 6 — Fix a couple of ordering issues

- **Check the distance-correlation metric before it is used to train anything.** Distance correlation
  is both what the projection head is trained to maximise *and* one of the numbers we report — so it
  needs its own correctness check, and that check is currently scheduled last. Fix: build the
  independent reference and the first checks in Phase 2, *before* any head is trained on it; make the
  check cover both the training version and the reported version; and, because the same quantity is
  trained on and reported, also report an independent quality number (like trustworthiness) so the
  claim is not circular. *(reorder + extend, S–M)*
- **Start the slow re-embedding job earlier.** It is the longest single task but only needs the branch
  setup and the source text — not the other shared pieces — so it can run alongside Phase 2 instead of
  after it. *(sequencing)*
- **When moving the seed-list file, keep the old path working.** The frozen plan and `CLAUDE.md` point
  to its current location; moving it would make those references wrong. Fix: keep a copy (or a link) at
  the old path, or add a one-line note to the plan in the same change; and add a test that the moved
  files actually ship inside both the wheel and the source package. *(low, S)*

### Theme 7 — Loose ends: extra analyses, other datasets, and the core claim

- **Decide, on the record, what happens to the analyses the plan promised but has no script for.** The
  pre-registered plan mentions a topic-attribution study (a 3-model rubric with an agreement measure),
  a real-data re-run of the out-of-sample filter, and an encoder-sensitivity study — none has a script,
  and the agreement measure is not implemented. Fix: one plan item that forces an explicit "we will do
  this" or "we are deferring this to a later version" for each, with a short note in the plan so a
  reader is not left expecting results that never appear. *(new item; L if kept, S if deferred)*
- **Decide about the two "pending" datasets** (corporate-sustainability and US-presidential). Their
  loaders currently just raise "not implemented." Fix: record a keep-or-defer decision for each. *(low)*
- **Add a direct test of REAP's core promise: that the consensus is unaffected by rotations and
  reflections.** The whole method rests on this, but nothing tests it head-on. Fix: apply random
  rotations and mirror-flips to the per-seed layouts and check the consensus distances (and the
  downstream agreement number) barely change. This is the cheapest, most convincing evidence for the
  paper's central idea. *(new item, S)*
- **Budget the writing itself.** The roadmap stops at "tables and figures" — but writing the methods
  section, the limitations, and doing the unfreeze-and-apply-corrections pass all take real time. Fix:
  add a final "write-up and unfreeze" phase. *(medium)*

### Theme 8 — Small determinism fixes

"Determinism" here means: run it twice with the same seed, get exactly the same answer.

- **Give the reporting step its own fixed randomness, separate from the model.** The metric's internal
  clustering has its own random start; feeding it the model's seed lets model randomness leak into the
  reported number. Fix: give it its own fixed seed. *(medium, S)*
- **Set the seed per fold, not once at the top.** A single seed at the top of the function can be
  "used up" by an early-stopping step, so later folds drift. Fix: derive each fold's seed from the base
  seed plus the fold number. *(low, S)*
- **Prove the head refactor changes nothing by default.** The plan refactors how the projection head is
  built. Fix: add a test that the default (MLP) path behaves identically before and after, and do the
  refactor and the seeding together. *(low, S)*

### Theme 9 — Minor corrections (cheap)

- Adding Python 3.13 to the test matrix is marked small, but some libraries (`numba`, `umap-learn`) may
  not have 3.13 wheels yet — check before committing to it; 3.10 is the priority.
- The roadmap's "Phase 3/4/5" labels clash with the protocol's own section numbers — rename or always
  qualify them, so nobody reads a false "these run in parallel."
- Both experiment branches are created before the shared pieces they use exist — fine, but land the
  shared pieces on `main` (or a shared branch both build on) so the two branches do not each rebuild
  them.
- Register the PyPI "trusted publisher" setting *before* the first release tag, or the first release
  will fail to publish.

---

## Updated order (only what changes)

The existing order is right; slot these in:

- **Before any Phase 3/4/5 experiment:** finish and commit the small protocol v1.5 amendment (Theme 1),
  with a TRACE note. This is now the gate on starting experiments.
- **Now (paperwork, no computing):** add the Korean-forest ARI correction to both correction lists and
  confirm the manuscripts' definition (Theme 3) — time-sensitive for Hye In.
- **Into Phase 2, before the experiment scripts:** the distance-correlation check (Theme 6), the
  canonical Korean-forest ARI definition + test (Theme 3), the Generalized Procrustes baseline
  (Theme 2), the checked `bundle.json` format + stop-on-missing combining (Theme 4), the
  rotation/reflection test (Theme 7), and the separate reporting-vs-model randomness (Theme 8).
- **Alongside Phase 2:** start the slow re-embedding job (Theme 6).
- **Attached to each experiment script and the combining script:** the quick tests (Theme 5).
- **New final phase:** write-up and unfreeze (Theme 7).

---

## Checked and dropped (so the review is honest both ways)

The review discarded these criticisms; they are listed so the plan's correctness is clear in both
directions.

**Turned out to be wrong:** the "best of N" baseline is leaky (it is chosen by cluster quality, not the
answer key); the main statistical test is underpowered at "N=3" (it uses 30 seeds); the record file
omits text-chunking details (those fields are already required); the "~May 2026" deadline is past with
no reset (the plan carries no calendar date); the effort tags are too sparse (every phase item already
has one).

**Already handled by the plan:** the cluster count is an uncontrolled confound (it is fixed once, in
advance); the out-of-sample Korean-forest corpus has no loader (it does); the torch/tensorflow
separation is ignored (the plan keeps them in separate jobs); the macOS number-range skip is
unprincipled (it is correctly split by test type); the path fixes come too late and strand the tests
(the plan sequences them with an honest skip path); the scripts hardcode result paths (they use the
repo root); the head refactor invalidates work done between refactor and seeding (nothing consumes such
work in that window); the public name list is neither trimmed nor deferred (both are in the plan); the
CI cleanup is front-loaded ahead of results (it changes no behaviour); the repo split reshapes things
while experiments still write results (that is exactly its two-stage design); the release governance is
set up before the facts are known (it is explicitly gated until results land).
