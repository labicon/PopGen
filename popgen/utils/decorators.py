import logging
import time
from functools import wraps


def track_time(label: str):
    """Decorator to log elapsed time for a method."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logging.info(f"{label} completed in {elapsed:.2f}s")
            return result, elapsed

        return wrapper

    return decorator
