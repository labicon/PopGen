"""Selection must ignore attempts that were never evaluated.

A diverged training run still leaves BenchMARL checkpoints written before the
crash, and a generation that failed every retry still leaves a policy.py. Both
used to be selectable with adherence=-inf, so a behavior whose every attempt
failed contributed a garbage teammate to the population.
"""

import json

import pytest

from popgen.cooperator.util import select_policy_refs_from_behavior_results


def _population(tmp_path, attempts):
    """Build a minimal population layout.

    `attempts` maps attempt index -> (kind, adherence or None for unevaluated).
    """
    root = tmp_path / "pop"
    run_dir = root / "agents" / "behavior_0" / "2026-01-01_00-00-00_000000"
    run_dir.mkdir(parents=True)

    for idx, (kind, adherence) in attempts.items():
        a = run_dir / f"attempt_{idx}"
        a.mkdir()
        if kind == "PolicyCode":
            (a / "policy.py").write_text("def policy(obs):\n    return 0\n")
        else:
            ck = a / "run" / "checkpoints"
            ck.mkdir(parents=True)
            (ck / "checkpoint_1000.pt").write_bytes(b"not-a-real-checkpoint")
        if adherence is not None:
            (a / "evaluation.json").write_text(json.dumps({"overall_metrics": {"adherence": adherence}}))

    meta = root / "metadata" / "2026-01-01_00-00-00_000000"
    meta.mkdir(parents=True)
    (meta / "summary.json").write_text(json.dumps({"runs": [{"behavior_idx": 0, "output_dir": str(run_dir)}]}))
    return root


@pytest.mark.parametrize("kind", ["RewardFn", "PolicyCode"])
def test_unevaluated_attempt_is_not_selected(tmp_path, kind):
    root = _population(tmp_path, {0: (kind, 0.6), 1: (kind, None)})

    refs = select_policy_refs_from_behavior_results(root)

    assert len(refs) == 1
    # attempt_1 has the higher index; without the filter it would win on the
    # (adherence, attempt_idx) tie-break once adherence is -inf... but more to
    # the point, it was never judged at all.
    assert refs[0].metadata["attempt_idx"] == 0
    assert refs[0].metadata["adherence"] == 0.6


@pytest.mark.parametrize("kind", ["RewardFn", "PolicyCode"])
def test_behavior_with_no_evaluated_attempt_is_dropped(tmp_path, kind):
    root = _population(tmp_path, {0: (kind, None), 1: (kind, None)})

    with pytest.raises(ValueError, match="No loadable teammate policy artifacts"):
        select_policy_refs_from_behavior_results(root)


def test_best_evaluated_attempt_still_wins(tmp_path):
    """The existing best-by-adherence behaviour is unchanged."""
    root = _population(tmp_path, {0: ("RewardFn", 0.2), 1: ("RewardFn", 1.0), 2: ("RewardFn", 0.4)})

    refs = select_policy_refs_from_behavior_results(root)

    assert len(refs) == 1
    assert refs[0].metadata["attempt_idx"] == 1
    assert refs[0].metadata["adherence"] == 1.0
