"""Per-worker log files for the population-generation pool.

Behavior runs execute in a ``pqdm``/``ProcessPoolExecutor`` pool. torchrl sets
the multiprocessing start method to ``spawn`` at import time, so each worker is
a fresh interpreter that re-runs the entry point's ``logging.basicConfig`` with
a stderr handler only -- Hydra's file handler lives in the parent and is *not*
inherited. Every per-attempt warning and traceback therefore reached the
terminal and nothing else.

Mirrors the ``configure_warning_filters(force=True)`` idiom used in the rollout
worker: set up process-local state inside the worker itself.
"""

import logging
import os

_LOG_FILENAME = "worker.log"
_CONFIGURED_PATHS: set[str] = set()


def configure_worker_logging(output_dir: str, *, level: int = logging.INFO) -> str | None:
    """Attach a FileHandler writing to ``<output_dir>/worker.log``.

    Idempotent per path within a process. Returns the log path, or None if the
    handler could not be installed (logging must never break a run).
    """
    try:
        log_path = os.path.join(output_dir, _LOG_FILENAME)
        if log_path in _CONFIGURED_PATHS:
            return log_path

        handler = logging.FileHandler(log_path)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(processName)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )

        root = logging.getLogger()
        root.addHandler(handler)
        if root.level > level:
            root.setLevel(level)

        _CONFIGURED_PATHS.add(log_path)
        return log_path
    except Exception:  # pragma: no cover - logging setup must not fail a run
        logging.warning("Could not attach worker log file in %s", output_dir, exc_info=True)
        return None
