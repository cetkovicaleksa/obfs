import pytest

from obfs import Adjacency
from obfs.bfs_engines import plaintext_bfs
from obfs.bfs_plain import bfs


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_plaintext_matches_reference(
    graph: str, request: pytest.FixtureRequest
) -> None:
    adjacency: Adjacency = request.getfixturevalue(graph)
    for source in range(len(adjacency)):
        assert plaintext_bfs(adjacency, source) == bfs(adjacency, source)


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_plaintext_burst_is_one_plus_degree(
    graph: str, request: pytest.FixtureRequest
) -> None:
    # Spec §4.12, PLAINTEXT_BFS step 2: the server sees vertex u and 1 + deg(u).
    adjacency: Adjacency = request.getfixturevalue(graph)
    for source in range(len(adjacency)):
        result = plaintext_bfs(adjacency, source)
        assert result.burst_lengths == [1 + len(adjacency[u]) for u in result.order]
