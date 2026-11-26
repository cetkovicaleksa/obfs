from pathlib import Path

import pytest

from obfs import Adjacency
from obfs.datasets import example

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def graphs_dir() -> Path:
    return FIXTURES_DIR / "graphs"


@pytest.fixture
def chain() -> Adjacency:
    adjacency, _ = example("chain")
    return adjacency


@pytest.fixture
def star() -> Adjacency:
    adjacency, _ = example("star")
    return adjacency


@pytest.fixture
def cycle() -> Adjacency:
    adjacency, _ = example("cycle")
    return adjacency
