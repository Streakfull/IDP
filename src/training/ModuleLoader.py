import importlib
import importlib.util
from pathlib import Path
import sys


def load_module_from_path(filepath: Path, class_name: str):
    module = _load_module(filepath)
    module_obj = getattr(module, class_name)
    try:
        return module_obj
    except TypeError as e:
        hint = ("The specific model config is passed to the constructor as an argument. The model needs"
                " to take a single argument in the constructor.")
        raise TypeError(f"{e}\n{hint}")


def _load_module(filepath: Path):
    module_name = filepath.name
    spec = importlib.util.spec_from_file_location(
        module_name, filepath.resolve())
    if spec is not None:
        _spec = spec
        module = importlib.util.module_from_spec(_spec)
        sys.modules[module_name] = module
        assert _spec.loader is not None
        loader = _spec.loader
        loader.exec_module(module)
        return module
    raise ImportError(f"Couldn't find a module under: {filepath}")
