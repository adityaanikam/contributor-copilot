import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CORPUS = ROOT / "tests" / "sample_corpus"


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts/ is not a package)."""
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def sample_corpus() -> Path:
    return SAMPLE_CORPUS
