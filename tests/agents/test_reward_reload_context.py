"""A RewardFn checkpoint must be reloaded under the scenario it was trained with.

`pp` and `pp_stag` both register the VMAS scenario name "simple_tag" but ship
different `semdiv_simple_tag` modules with different observation widths. The
reload path used to hardcode `pp`'s module, so a pp_stag checkpoint was rebuilt
into a 30-wide model and died with "Failed to update 'weight'" -- after its full
41-minute training had already completed, and then again on every retry.
"""

import importlib
from contextlib import contextmanager

import pytest

from popgen.agents.reward_policy import _reload_scenarios


def test_pp_and_pp_stag_ship_different_scenarios():
    """The precondition that makes a hardcoded scenario wrong."""
    pp = importlib.import_module("popgen.envs.pp.semdiv_simple_tag")
    pp_stag = importlib.import_module("popgen.envs.pp_stag.semdiv_simple_tag")
    assert pp is not pp_stag


@pytest.mark.parametrize("env_name", ["pp", "pp_stag"])
def test_env_reload_context_installs_its_own_scenario(env_name):
    """Each env's reload context must patch in that env's scenario, not a sibling's."""
    import vmas

    env_utils = importlib.import_module(f"popgen.envs.{env_name}.utils")
    own_scenario = importlib.import_module(f"popgen.envs.{env_name}.semdiv_simple_tag").Scenario

    captured = {}
    original = vmas.make_env
    try:
        vmas.make_env = lambda scenario, *a, **k: captured.setdefault("scenario", scenario)
        with env_utils.get_cooperator_reload_context():
            vmas.make_env("simple_tag")
    finally:
        vmas.make_env = original

    assert isinstance(captured["scenario"], own_scenario), (
        f"{env_name} reload context installed {type(captured['scenario'])}, not its own Scenario"
    )


def test_reload_scenarios_prefers_the_supplied_context():
    used = []

    @contextmanager
    def fake_context():
        used.append(True)
        yield

    with _reload_scenarios(fake_context):
        pass
    assert used == [True]


def test_reload_scenarios_falls_back_when_env_has_no_context():
    """Envs without a reload context still get the legacy behaviour."""
    import vmas

    original = vmas.make_env
    with _reload_scenarios(None):
        assert vmas.make_env is not original
    assert vmas.make_env is original
