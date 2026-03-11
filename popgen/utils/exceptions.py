import traceback


def format_exc_last_frame(exc: BaseException) -> str:
    """Return only the deepest traceback frame plus the exception message."""
    tb = exc.__traceback__
    if tb is None:
        return f"{type(exc).__name__}: {exc}"
    last = traceback.extract_tb(tb)[-1]
    return f'File "{last.filename}", line {last.lineno}, in {last.name}\n    {last.line}\n{type(exc).__name__}: {exc}'
