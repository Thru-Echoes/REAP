"""Docstring pins for the two registered distance-correlation recipe names.

The Korean-forest ARI confusion came from one everyday name quietly covering
several different calculations. "Distance correlation" has the same exposure:
the reported metric and the training loss are two different calculations, and
protocol v1.5 clause (m) cites each recipe's docstring wording verbatim. These
tests fail if either production docstring stops carrying the load-bearing
recipe facts:

- its registered recipe id (``distance_correlation`` / ``dist_corr_loss``),
- the explicit "not Székely distance correlation" disclaimer,
- the defining calculation detail (condensed vectors vs diagonal-included
  full matrix), and
- the cross-reference to the *other* recipe, so no reader can conflate them.

Wording may be edited freely; these pins only assert the tokens whose loss
would re-open the one-name-two-calculations hole.
"""

from __future__ import annotations

from reap.evaluation import compute_distance_correlation
from reap.projection import compute_projection_loss


def test_reported_metric_docstring_pins_condensed_recipe():
    """compute_distance_correlation documents recipe id, variant, disclaimer."""
    doc = compute_distance_correlation.__doc__ or ""
    assert "distance_correlation" in doc  # registered recipe id
    assert "condensed" in doc  # the defining calculation detail
    assert "Székely" in doc  # explicit not-Székely disclaimer
    assert "adapted" in doc  # metric kind tag per the metrics catalog
    assert "dist_corr_loss" in doc  # cross-reference to the sibling recipe


def test_loss_docstring_pins_full_matrix_recipe():
    """compute_projection_loss documents the loss recipe and its freeze."""
    doc = compute_projection_loss.__doc__ or ""
    assert "dist_corr_loss" in doc  # registered recipe id
    assert "diagonal" in doc  # the defining calculation detail
    assert "Székely" in doc  # explicit not-Székely disclaimer
    assert "distance_correlation" in doc  # cross-reference to the sibling recipe
    assert "byte-identical" in doc  # calibration-freeze note
