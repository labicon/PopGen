import os
import traceback

_TRAINING_LOG_TAIL_LINES = 40


def format_exc_last_frame(exc: BaseException) -> str:
    """Return only the deepest traceback frame plus the exception message."""
    tb = exc.__traceback__
    if tb is None:
        return f"{type(exc).__name__}: {exc}"
    last = traceback.extract_tb(tb)[-1]
    return f'File "{last.filename}", line {last.lineno}, in {last.name}\n    {last.line}\n{type(exc).__name__}: {exc}'


def _read_log_tail(path: str | None, n_lines: int = _TRAINING_LOG_TAIL_LINES) -> str:
    """Return the last *n_lines* of *path*, or "" if it cannot be read."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return ""
    return "".join(lines[-n_lines:]).strip()


class TrainingFailure(Exception):
    """RL training itself failed, as opposed to generating or importing the reward.

    Training costs tens of minutes, so the pipeline charges these to a separate,
    much smaller retry budget than cheap generation errors.

    The instance is *returned* rather than raised, so it never carries a
    traceback of its own. The message is therefore rendered at construction
    time, while the original exception's traceback is still live.
    """

    def __init__(self, original: BaseException, elapsed: float = 0.0, training_log: str | None = None):
        self.original = original
        self.elapsed = float(elapsed)
        self.training_log = training_log

        message = format_exc_last_frame(original)
        # The deepest BenchMARL frame for a NaN divergence says little about the
        # reward that caused it. The tail of the training log carries the
        # entropy/clip_fraction trace, which is the signal the model needs to
        # write a better-scaled reward next attempt.
        tail = _read_log_tail(training_log)
        if tail:
            message = f"{message}\n\nLast lines of the training log ({training_log}):\n{tail}"
        super().__init__(message)
