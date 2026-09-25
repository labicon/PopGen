"""Tests for dynamic loading of generated code artifacts."""

import textwrap

from popgen.utils.imports import import_all_functions_from_file, import_function_from_file


def _write(path, value):
    path.write_text(textwrap.dedent(f"""
        def compute_reward(self, agent):
            return {value}

        def policy(obs):
            return {value}
    """))


def test_rewriting_same_path_reloads_new_code(tmp_path):
    """A regenerated artifact at the same path must not be served from cache.

    The reward/policy pipelines rewrite ``reward.py`` / ``policy.py`` in place
    on every retry. Caching on path alone silently re-ran the first version,
    so each retry retrained an identical failing reward.
    """
    artifact = tmp_path / "reward.py"

    _write(artifact, 1.0)
    assert import_function_from_file(str(artifact), "compute_reward")(None, None) == 1.0

    _write(artifact, 999.0)
    assert import_function_from_file(str(artifact), "compute_reward")(None, None) == 999.0


def test_rewriting_same_path_reloads_for_all_functions(tmp_path):
    """The policy-code path (import_all_functions_from_file) must reload too."""
    artifact = tmp_path / "policy.py"

    _write(artifact, 1.0)
    assert import_all_functions_from_file(str(artifact))["policy"](None) == 1.0

    _write(artifact, 999.0)
    assert import_all_functions_from_file(str(artifact))["policy"](None) == 999.0


def test_unchanged_file_reuses_cached_module(tmp_path):
    """An unmodified file still returns the same module object.

    Callers rely on a stable ``__module__`` for pickling, so repeated loads of
    unchanged source must not produce a fresh module each time.
    """
    artifact = tmp_path / "reward.py"
    _write(artifact, 1.0)

    first = import_function_from_file(str(artifact), "compute_reward")
    second = import_function_from_file(str(artifact), "compute_reward")
    assert first is second


def test_distinct_paths_are_independent(tmp_path):
    """Two attempt directories each keep their own version."""
    a = tmp_path / "attempt_0" / "reward.py"
    b = tmp_path / "attempt_1" / "reward.py"
    a.parent.mkdir()
    b.parent.mkdir()
    _write(a, 1.0)
    _write(b, 2.0)

    assert import_function_from_file(str(a), "compute_reward")(None, None) == 1.0
    assert import_function_from_file(str(b), "compute_reward")(None, None) == 2.0
    assert import_function_from_file(str(a), "compute_reward")(None, None) == 1.0
