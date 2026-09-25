"""Reload a BenchMARL experiment from a checkpoint, tolerating unloadable callbacks.

``Experiment.reload_from_file`` unpickles ``config.pkl`` field by field, and the last
field is the training callbacks. Those callbacks routinely reference modules that only
existed in the training process:

* ``popgen._dynamic.<md5>`` -- dynamically loaded generated policy/reward files.
* bare top-level names such as ``callback`` -- a script-local module that was importable
  only because Hydra ran with that script's directory on ``sys.path``. MEP population
  members are the live example: their callbacks are ``callback.SelfPlayTeammateCallback``
  and ``callback.PopulationEntropyRewardCallback`` from ``scripts/mep/callback.py``, so
  the same checkpoint reloads fine from ``scripts/mep/`` and dies with
  ``ModuleNotFoundError: No module named 'callback'`` from anywhere else.

Callbacks are training-time hooks and carry nothing the policy needs, so for inference
the right answer is to drop them rather than to make every caller reproduce the training
process's ``sys.path``.
"""

from __future__ import annotations

import inspect
import logging
import pickle
import sys
import types
from pathlib import Path
from typing import Any

# Raised when a pickled callback references a module or attribute that no longer exists.
_CALLBACK_LOAD_ERRORS = (pickle.UnpicklingError, EOFError, ModuleNotFoundError, AttributeError, ImportError)
# Raised when a pickled algorithm config carries a field the current class no longer takes.
_RELOAD_ERRORS = _CALLBACK_LOAD_ERRORS + (TypeError,)

_CONFIG_FIELDS = (
    "task",
    "task_config",
    "algorithm_config",
    "model_config",
    "seed",
    "experiment_config",
    "critic_model_config",
)


class _StubCallable:
    """Stands in for a callback class that can no longer be imported."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None


def register_dynamic_stubs() -> type[pickle.Unpickler]:
    """Return an Unpickler that fabricates stubs for ``popgen._dynamic.*`` modules."""
    if "popgen._dynamic" not in sys.modules:
        pkg = types.ModuleType("popgen._dynamic")
        pkg.__package__ = "popgen._dynamic"
        pkg.__path__ = []
        sys.modules["popgen._dynamic"] = pkg

    class _StubUnpickler(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if module.startswith("popgen._dynamic."):
                if module not in sys.modules:
                    stub = types.ModuleType(module)
                    stub.__package__ = "popgen._dynamic"
                    sys.modules[module] = stub
                mod = sys.modules[module]
                if not hasattr(mod, name):
                    setattr(mod, name, _StubCallable())
                return getattr(mod, name)
            return super().find_class(module, name)

    return _StubUnpickler


def _drop_unsupported_algorithm_fields(algorithm_config: Any) -> list[str]:
    """Remove config fields the algorithm class no longer accepts, returning their names.

    ``AlgorithmConfig.get_algorithm`` splats ``self.__dict__`` into the algorithm's
    constructor, so a field that has since been removed makes the checkpoint unloadable:
    ``Algorithm.__init__() got an unexpected keyword argument 'normalize_advantage'``.
    The run itself is fine -- only the constructor signature moved on -- and none of these
    fields affect a forward pass, so for inference they are dropped rather than lost.
    """
    try:
        cls = algorithm_config.associated_class()
    except Exception:
        return []

    accepted: set[str] = set()
    for klass in inspect.getmro(cls):
        try:
            signature = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        if any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values()):
            accepted |= {p.name for p in signature.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        else:
            accepted |= set(signature.parameters)

    dropped = sorted(set(algorithm_config.__dict__) - accepted - {"self"})
    for field in dropped:
        delattr(algorithm_config, field)
    return dropped


def reload_experiment(Experiment, checkpoint: Path, experiment_patch: dict[str, Any]):
    """Reload *checkpoint*, tolerating unloadable callbacks and stale config fields."""
    try:
        return Experiment.reload_from_file(str(checkpoint), experiment_patch=experiment_patch)
    except _RELOAD_ERRORS as exc:
        config_file = Path(checkpoint).parent.parent.resolve() / "config.pkl"
        logging.warning(
            "Falling back to callback-free reload for %s after config.pkl load failed: %s: %s",
            config_file,
            type(exc).__name__,
            exc,
        )

        with open(config_file, "rb") as f:
            fields = {name: pickle.load(f) for name in _CONFIG_FIELDS}
            try:
                callbacks = pickle.load(f)
            except _CALLBACK_LOAD_ERRORS as callback_exc:
                logging.warning(
                    "Ignoring unloadable callbacks in %s: %s: %s",
                    config_file,
                    type(callback_exc).__name__,
                    callback_exc,
                )
                callbacks = []

        dropped = _drop_unsupported_algorithm_fields(fields["algorithm_config"])
        if dropped:
            logging.warning(
                "Dropping algorithm config field(s) %s from %s: the current %s no longer accepts them. They do not affect inference.",
                ", ".join(dropped),
                config_file,
                type(fields["algorithm_config"]).__name__,
            )

        task = fields["task"]
        task.config = fields["task_config"]
        experiment_config = fields["experiment_config"]
        experiment_config.restore_file = str(checkpoint)
        for key, value in experiment_patch.items():
            if not hasattr(experiment_config, key):
                raise ValueError(f"Experiment config does not have attribute {key}")
            setattr(experiment_config, key, value)

        experiment = Experiment(
            task=task,
            algorithm_config=fields["algorithm_config"],
            model_config=fields["model_config"],
            seed=fields["seed"],
            config=experiment_config,
            callbacks=callbacks,
            critic_model_config=fields["critic_model_config"],
        )
        logging.info("Reloaded experiment %s from %s (callbacks dropped).", experiment.name, checkpoint)
        return experiment
