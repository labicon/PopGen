import hashlib
import importlib.util
import os
import sys


def _file_stamp(abs_path: str) -> str | None:
    """Return a content signature for *abs_path*, or None if unreadable.

    Content-hashed rather than mtime-based: generated artifacts are rewritten
    in place within seconds on a retry, and filesystem mtime granularity is
    coarse enough to miss that. These files are a few KB, so hashing is cheap.
    """
    try:
        with open(abs_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return None


def _module_matches_file(module, abs_path: str) -> bool:
    """Return True when *module* is a real, up-to-date import of *abs_path*.

    Generated artifacts (``reward.py``, ``policy.py``) are rewritten in place on
    every retry, so matching the path alone would hand back the previous
    version's module and silently re-run stale code.
    """
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    if os.path.abspath(module_file) != abs_path:
        return False
    stamp = getattr(module, "__popgen_file_stamp__", None)
    return stamp is not None and stamp == _file_stamp(abs_path)


def _load_module_from_file(file_path: str):
    """Load a module from *file_path*, registering it in ``sys.modules``.

    A stable, unique module name is derived from the absolute path so that
    ``pickle`` can locate functions by their ``__module__`` attribute.
    If a placeholder module already exists under that name (for example a
    stub inserted while unpickling checkpoints), it is replaced.
    """
    abs_path = os.path.abspath(file_path)
    # Unique but human-readable: "popgen._dynamic.<hex>" where hex is the MD5
    # of the absolute path.
    module_name = "popgen._dynamic." + hashlib.md5(abs_path.encode()).hexdigest()

    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        if _module_matches_file(existing_module, abs_path):
            return existing_module
        # Placeholder, or the file changed on disk since it was loaded;
        # reload from source.
        del sys.modules[module_name]

    stamp = _file_stamp(abs_path)
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {abs_path}")
    module = importlib.util.module_from_spec(spec)
    # Record the source signature so a later rewrite of the same path is
    # detected instead of being served from sys.modules.
    module.__popgen_file_stamp__ = stamp
    # Register *before* exec so that recursive imports resolve correctly.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_entrypoint(path):
    module_name, attr = path.split(":")
    module = importlib.import_module(module_name)
    obj = module
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def import_function_from_file(file_path: str, function_name: str):
    """Dynamically import a function from a given file."""
    module = _load_module_from_file(file_path)
    return getattr(module, function_name)


def import_all_functions_from_file(file_path: str, regex: str = None) -> dict:
    """Dynamically import all functions from a given file."""
    module = _load_module_from_file(file_path)
    functions = {name: func for name, func in module.__dict__.items() if callable(func)}
    # If a regex is provided, filter functions by it
    if regex is not None:
        import re

        pattern = re.compile(regex)
        functions = {name: func for name, func in functions.items() if pattern.match(name)}
    return functions
