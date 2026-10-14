from __future__ import annotations

import numpy as np
import pytest

from fmri_sae_correspondence.alignment import (
    TrBin,
    WordTiming,
    assign_words_to_trs,
    build_token_to_word_index,
    group_tokens_into_words,
    infer_word_start_marker,
    validate_tr_alignment,
    validate_word_alignment,
)
from fmri_sae_correspondence.feature_banks import compute_feature_stats, contiguous_fold_rows
from fmri_sae_correspondence.modeling import (
    benjamini_hochberg,
    feature_groups,
    finite_rows,
    scalar_pearson,
    scalar_spearman,
    topk_jaccard,
    within_layer_identity_null,
)


@pytest.mark.parametrize(
    ("model_id", "tokens", "expected"),
    [
        ("google/gemma-2-2b", ["▁hello"], "▁"),
        ("meta-llama/Llama-3.1-8B", ["Ġhello"], "Ġ"),
        ("custom/model", ["plain"], None),
    ],
)
def test_infer_word_start_marker(model_id: str, tokens: list[str], expected: str | None) -> None:
    assert infer_word_start_marker(tokens, model_id) == expected


def test_group_tokens_into_words_preserves_token_indices() -> None:
    groups = group_tokens_into_words(["▁hello", "!", "▁world"], "▁")
    assert [(group.text, group.token_indices) for group in groups] == [
        ("hello!", (0, 1)),
        ("world", (2,)),
    ]


def test_build_token_to_word_index_maps_subwords() -> None:
    groups = group_tokens_into_words(["▁break", "ing", "▁now"], "▁")
    assert build_token_to_word_index(groups) == {0: 0, 1: 0, 2: 1}


def test_assign_words_to_trs_uses_half_open_bins() -> None:
    words = [WordTiming("one", 0.0, 1.0), WordTiming("two", 1.0, 2.0)]
    bins = [TrBin(0, 0.0, 1.0, "one"), TrBin(1, 1.0, 2.0, "two")]
    assert assign_words_to_trs(words, bins, 0.0) == [0, 1]


def test_assign_words_to_trs_accepts_final_endpoint() -> None:
    words = [WordTiming("last", 1.5, 2.5)]
    bins = [TrBin(0, 0.0, 1.0, ""), TrBin(1, 1.0, 2.0, "last")]
    assert assign_words_to_trs(words, bins, 0.0) == [1]


def test_assign_words_to_trs_rejects_outside_timing() -> None:
    with pytest.raises(ValueError, match="could not assign"):
        assign_words_to_trs([WordTiming("late", 3.0, 4.0)], [TrBin(0, 0.0, 1.0, "")], 0.0)


def test_validate_word_alignment_accepts_normalized_punctuation() -> None:
    groups = group_tokens_into_words(["▁hello", "!"], "▁")
    validate_word_alignment(groups, [WordTiming("Hello!", 0.0, 0.5)], "Hello!")


def test_validate_word_alignment_rejects_mismatch() -> None:
    groups = group_tokens_into_words(["▁hello"], "▁")
    with pytest.raises(ValueError, match="mismatches"):
        validate_word_alignment(groups, [WordTiming("goodbye", 0.0, 0.5)], "goodbye")


def test_validate_tr_alignment_accepts_matching_bins() -> None:
    words = [WordTiming("one", 0.0, 0.5), WordTiming("two", 1.0, 1.5)]
    bins = [TrBin(0, 0.0, 1.0, "one"), TrBin(1, 1.0, 2.0, "two")]
    validate_tr_alignment(words, [0, 1], bins, 0.0)


def test_feature_groups_keep_all_lags_together() -> None:
    groups = feature_groups(3, 2)
    assert [group.tolist() for group in groups] == [[0, 1], [2, 3], [4, 5]]


def test_finite_rows_excludes_any_nonfinite_target() -> None:
    values = np.array([[1.0, 2.0], [np.nan, 1.0], [2.0, np.inf], [3.0, 4.0]])
    np.testing.assert_array_equal(finite_rows(values), np.array([0, 3]))


def test_scalar_correlations_handle_identical_rankings() -> None:
    values = np.array([4.0, 1.0, 3.0, 2.0])
    assert scalar_pearson(values, values) == pytest.approx(1.0)
    assert scalar_spearman(values, values) == pytest.approx(1.0)


def test_topk_jaccard_compares_feature_identity() -> None:
    assert topk_jaccard(np.array([4, 3, 2, 1]), np.array([4, 3, 1, 2]), 2) == 1.0


def test_benjamini_hochberg_is_monotone_in_rank() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.8])
    assert np.all((0.0 <= adjusted) & (adjusted <= 1.0))
    assert adjusted[0] <= adjusted[2] <= adjusted[1] <= adjusted[3]


def test_identity_null_is_deterministic_and_within_layer() -> None:
    brain = np.array([0.4, 0.3, 0.2, 0.1])
    lm = brain.copy()
    names = np.array(["layer4:feature1", "layer4:feature2", "layer8:feature1", "layer8:feature2"])
    first = within_layer_identity_null(brain, lm, names, n_permutations=20, seed=7)
    second = within_layer_identity_null(brain, lm, names, n_permutations=20, seed=7)
    np.testing.assert_array_equal(first["null"], second["null"])


def test_feature_stats_are_outcome_free_and_thresholded() -> None:
    values = np.array([[[1.0, 0.0], [1.0, 2.0], [0.0, 2.0], [0.0, 0.0]]])
    stats = compute_feature_stats(values, [0, 1, 2, 3], min_active_trs=2, max_active_fraction=0.75)
    np.testing.assert_array_equal(stats["active_count"][0], np.array([2, 2]))
    np.testing.assert_array_equal(stats["eligible"][0], np.array([0, 1]))


def test_contiguous_folds_purge_held_neighbors() -> None:
    for train, held in contiguous_fold_rows(20, 5, purge_adjacent_trs=2):
        for row in held:
            assert not np.any(np.abs(train - row) <= 2)
