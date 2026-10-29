import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

from jparacrawl_filtering.config import load_config

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo", ROOT / "scripts/synthetic_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture
def config():
    return deepcopy(load_config(ROOT / "configs/default.yaml"))


@pytest.fixture
def lid():
    return demo.SyntheticLanguageID()


@pytest.fixture
def prepared(tmp_path):
    return demo.prepare(tmp_path / "work")
