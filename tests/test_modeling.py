from __future__ import annotations

import numpy as np

from fmri_sae_correspondence import modeling


def test_block_labels_are_contiguous_and_complete() -> None:
    labels = modeling.block_labels(10, 3)
    np.testing.assert_array_equal(labels, np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2]))


def test_purged_training_positions_removes_held_block_and_neighbors() -> None:
    positions = modeling.purged_training_positions(
        12,
        np.array([5, 6]),
        allowed_positions=np.arange(1, 11),
        purge_adjacent=2,
    )
    np.testing.assert_array_equal(positions, np.array([1, 2, 9, 10]))


def test_lag_expand_preserves_feature_grouping() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    expanded = modeling.lag_expand(values, [0, 1])
    expected = np.array(
        [
            [1.0, 0.0, 10.0, 0.0],
            [2.0, 1.0, 20.0, 10.0],
            [3.0, 2.0, 30.0, 20.0],
        ]
    )
    np.testing.assert_array_equal(expanded, expected)


def test_correspondence_recovers_identical_rankings() -> None:
    values = np.array([0.4, 0.1, 0.3, 0.2])
    result = modeling.correspondence(values, values, top_k=2)
    assert result["spearman"] == 1.0
    assert result["topk_jaccard"] == 1.0
