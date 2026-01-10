import importlib.util

def import_function_from_file(file_path: str, function_name: str):
    """Dynamically import a function from a given file."""
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)