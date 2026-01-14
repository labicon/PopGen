import importlib.util


def import_function_from_file(file_path: str, function_name: str):
    """Dynamically import a function from a given file."""
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def import_all_functions_from_file(file_path: str, regex: str = None) -> dict:
    """Dynamically import all functions from a given file."""
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    functions = {name: func for name, func in module.__dict__.items() if callable(func)}
    # If a regex is provided, filter functions by it
    if regex is not None:
        import re

        pattern = re.compile(regex)
        functions = {name: func for name, func in functions.items() if pattern.match(name)}
    return functions
