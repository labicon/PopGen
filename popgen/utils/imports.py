import hashlib
import importlib.util
import os
import sys


def _module_matches_file(module, abs_path: str) -> bool:
    """Return True when *module* is a real import of *abs_path*."""
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str):
        return False
    return os.path.abspath(module_file) == abs_path


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
        # Placeholder module is stale for this file; reload from source.
        del sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {abs_path}")
    module = importlib.util.module_from_spec(spec)
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
