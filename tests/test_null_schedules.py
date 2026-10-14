from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from fmri_sae_correspondence import nulls
from fmri_sae_correspondence.config import default_config_path
from fmri_sae_correspondence.integrity import load_spec


@pytest.mark.parametrize("part", ["feature", "within-grouped", "cross-grouped"])
def test_runner_shares_null_schedules_across_models(monkeypatch, tmp_path, part):
    config_path = default_config_path()
    spec = load_spec(config_path)
    calls = []

    def record(config_path, spec, model, *condition, n_shifts, seed, force):
        calls.append((model, condition, seed))
        return {"complete": True}

    for name in (
        "_within_temporal_null",
        "_cross_temporal_null",
        "_grouped_temporal_null_within",
        "_grouped_temporal_null_cross",
    ):
        monkeypatch.setattr(nulls, name, record)
    monkeypatch.setattr(nulls, "_artifact_root", lambda *args: tmp_path)
    args = SimpleNamespace(
        models=list(spec["models"]),
        variants=["source_story_variance", "source_story_prevalence"],
        sizes=[128, 256],
        grouped_thresholds=[0.7, 0.8],
        parts=part,
    )
    nulls.run(config_path=config_path, spec=spec, args=args)

    conditions = {condition for _, condition, _ in calls}
    assert conditions
    for condition in conditions:
        selected = [(model, seed) for model, key, seed in calls if key == condition]
        assert {model for model, _ in selected} == set(spec["models"])
        assert len({seed for _, seed in selected}) == 1
    # Direction, bank, size, and grouping threshold still define separate draws.
    assert len({seed for _, _, seed in calls}) == len(conditions)

    if part == "feature":
        reference_hashes = {
            ("shapesphysical", "shapessocial"): "622d407d0cebc0a516dbb33ee3775cfd9cccea7054d30fb39ff433b2432eed65",
            ("shapessocial", "shapesphysical"): "dae07269246892857056d99a3225389faf24146ad2f5b80e80e1e0859e7b66b1",
        }
        for direction, expected_hash in reference_hashes.items():
            condition = (*direction, "source_story_variance", 256)
            seed = next(seed for _, key, seed in calls if key == condition)
            source, target = nulls.fixed_independent_transfer_offsets(
                spec["stories"][direction[0]]["expected_trs"],
                spec["stories"][direction[1]]["expected_trs"],
                199, 10, seed,
            )
            # Digest of the paired offset schedule used by the reported K=256 null.
            encoded = np.column_stack([source, target]).astype("<i8").tobytes()
            assert hashlib.sha256(encoded).hexdigest() == expected_hash


def test_transfer_offsets_are_unique_nontrivial_and_differ_between_targets():
    source, target = nulls.fixed_independent_transfer_offsets(303, 305, 199, 10, 7)
    for offsets, length in [(source, 303), (target, 305)]:
        assert len(np.unique(offsets)) == 199
        assert np.all(np.minimum(offsets, length - offsets) >= 10)
    assert np.all(source != target)


def test_offset_generator_rejects_more_draws_than_distinct_valid_shifts():
    with pytest.raises(ValueError, match="unique shifts"):
        nulls.fixed_shift_offsets(25, 7, 10, 7)
