from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def graphs_dir() -> Path:
    return FIXTURES_DIR / "graphs"
