from pathlib import Path

import pytest

from obfs import Adjacency
from obfs.graph_loader import load_graph

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def graphs_dir() -> Path:
    return FIXTURES_DIR / "graphs"

@pytest.fixture
def chain(graphs_dir: Path) -> Adjacency:
    adjacency, _ = load_graph(graphs_dir / "chain.txt", "facebook")
    return adjacency


@pytest.fixture
def star(graphs_dir: Path) -> Adjacency:
    adjacency, _ = load_graph(graphs_dir / "star.txt", "facebook")
    return adjacency


@pytest.fixture
def cycle(graphs_dir: Path) -> Adjacency:
    adjacency, _ = load_graph(graphs_dir / "cycle.txt", "facebook")
    return adjacency