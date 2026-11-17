"""Label discovery for the corporate and presidential climate-language case study.

Before a codebook for this case study is frozen, its categories are found from
the passages themselves: a passage pool is built (D0), a stratified discovery
sample is drawn (D1), language models code each sampled passage with free
labels and no codebook (D2), the free labels are grouped into candidate
codebooks (D3), and the candidate codebooks are crossed against a fixed list of
deductive frames (D4). A person then reviews the crosswalk and decides the
version-0 codebook.

Hard rule for every module here: discovery never reads REAP topics, maps,
or clusters for this corpus, so the labels can later serve as ground truth
for REAP without circularity.

Modules: ``passages`` (chunking, keywords, verbatim checks), ``sources``
(read-only readers for the White House tree and the corporate extractor
manifest), ``records`` (the ``bundle.json`` record file), ``pool`` (D0),
``sample`` (D1), ``coding`` (D2), ``codebooks`` (D3), ``crosswalk`` (D4),
``files`` (atomic JSON/JSONL writers).

This package lives under ``scripts/`` because it is research-specific; the
``reap`` package core stays research-agnostic.
"""
