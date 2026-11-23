import networkx as nx
import pytest

from obfs import Adjacency, BfsResult
from obfs.bfs_plain import bfs


def test_chain_from_vertex_2(chain: Adjacency) -> None:
    # Chain 1-2-3-4-5 is loaded as 0..4, so source 2 is vertex 1.
    assert bfs(chain, 1) == BfsResult(
        dist={1: 0, 0: 1, 2: 1, 3: 2, 4: 3},
        order=[1, 0, 2, 3, 4],
        burst_lengths=[3, 2, 3, 3, 2],
    )


# Spec §2.1 table and Appendix B.1, with sources 1..5 loaded as 0..4.
@pytest.mark.parametrize(
    ("source", "order", "burst_lengths"),
    [
        (0, [0, 1, 2, 3, 4], [2, 3, 3, 3, 2]),
        (1, [1, 0, 2, 3, 4], [3, 2, 3, 3, 2]),
        (2, [2, 1, 3, 0, 4], [3, 3, 3, 2, 2]),
        (3, [3, 2, 4, 1, 0], [3, 3, 2, 3, 2]),
        (4, [4, 3, 2, 1, 0], [2, 3, 3, 3, 2]),
    ],
)
def test_chain_matches_spec_table(
    chain: Adjacency, source: int, order: list[int], burst_lengths: list[int]
) -> None:
    result = bfs(chain, source)
    assert result.order == order
    assert result.burst_lengths == burst_lengths


def test_star_matches_appendix_b2(star: Adjacency) -> None:
    assert bfs(star, 0).burst_lengths == [5, 2, 2, 2, 2]
    for leaf in range(1, 5):
        assert bfs(star, leaf).burst_lengths == [2, 5, 2, 2, 2]


def test_cycle_matches_appendix_b3(cycle: Adjacency) -> None:
    for source in range(5):
        assert bfs(cycle, source).burst_lengths == [3, 3, 3, 3, 3]


def test_same_source_gives_same_result(chain: Adjacency) -> None:
    assert bfs(chain, 2) == bfs(chain, 2)


@pytest.mark.parametrize("source", [-1, 5])
def test_invalid_source_is_rejected(chain: Adjacency, source: int) -> None:
    with pytest.raises(ValueError):
        bfs(chain, source)


def connected_graphs() -> list[nx.Graph]:
    graphs = [
        nx.connected_watts_strogatz_graph(50, 4, 0.3, seed=seed) for seed in range(5)
    ]
    graphs.append(nx.karate_club_graph())
    return graphs


@pytest.mark.parametrize("graph", connected_graphs())
def test_every_source_matches_networkx(graph: nx.Graph) -> None:
    # Vertices of these graphs are already 0..n-1.
    adjacency = [sorted(graph[u]) for u in range(graph.number_of_nodes())]
    for source in range(len(adjacency)):
        result = bfs(adjacency, source)
        assert result.dist == nx.single_source_shortest_path_length(graph, source)
        assert (
            sum(burst - 1 for burst in result.burst_lengths)
            == 2 * graph.number_of_edges()
        )
