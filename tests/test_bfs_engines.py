import random

import networkx as nx
import pytest

from obfs import Adjacency
from obfs.bfs_engines import mitigated_oram_bfs, naive_oram_bfs, plaintext_bfs
from obfs.bfs_plain import bfs
from obfs.block_codec import encode_graph
from obfs.path_oram import PathORAM


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


SEED = 2027


def make_oram(
    adjacency: Adjacency, dummy_pool_size: int = 4, debug: bool = True
) -> PathORAM:
    blocks = encode_graph(adjacency, dummy_pool_size)
    return PathORAM(blocks, random.Random(SEED), debug=debug)


def random_graph() -> Adjacency:
    graph = nx.connected_watts_strogatz_graph(30, 4, 0.3, seed=SEED)
    return [sorted(graph[u]) for u in range(graph.number_of_nodes())]


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_naive_matches_reference(graph: str, request: pytest.FixtureRequest) -> None:
    # Spec §6.1, C5: identical distances and order. With ORAM the bursts are
    # the same too, 1 + deg(u) (Spec §4.12, NAIVE_ORAM_BFS step 2).
    adjacency: Adjacency = request.getfixturevalue(graph)
    oram = make_oram(adjacency)
    for source in range(len(adjacency)):
        assert naive_oram_bfs(oram, source) == bfs(adjacency, source)


def test_naive_matches_reference_on_random_graph() -> None:
    adjacency = random_graph()
    oram = make_oram(adjacency, debug=False)
    for source in range(0, len(adjacency), 7):
        assert naive_oram_bfs(oram, source) == bfs(adjacency, source)
    oram.check_invariant()


def test_naive_burst_is_logical_access_count(star: Adjacency) -> None:
    # Spec §2.6: each logical access is one read_path and one write_path, so
    # the server sees 2 * sum(bursts) path events.
    oram = make_oram(star)
    result = naive_oram_bfs(oram, 0)
    events = oram.server.events
    assert len(events) == 2 * sum(result.burst_lengths)
    assert [kind for kind, _ in events[::2]] == ["read_path"] * sum(
        result.burst_lengths
    )
    assert [kind for kind, _ in events[1::2]] == ["write_path"] * sum(
        result.burst_lengths
    )


def test_naive_rejects_source_that_is_not_a_vertex(star: Adjacency) -> None:
    # Block 5 is the first slot, not a header.
    with pytest.raises(ValueError):
        naive_oram_bfs(make_oram(star), 5)


def dummy_pool(adjacency: Adjacency, dummy_pool_size: int = 4) -> range:
    # Spec §4.11: the dummies follow the |V| headers and 2|E| slots.
    start = len(adjacency) + sum(len(neighbors) for neighbors in adjacency)
    return range(start, start + dummy_pool_size)


def max_degree(adjacency: Adjacency) -> int:
    return max(len(neighbors) for neighbors in adjacency)


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_mitigated_matches_reference(
    graph: str, request: pytest.FixtureRequest
) -> None:
    # Spec §4.14: same distances and order as plaintext, every burst 1 + B.
    adjacency: Adjacency = request.getfixturevalue(graph)
    budget = max_degree(adjacency)
    oram = make_oram(adjacency)
    for source in range(len(adjacency)):
        result = mitigated_oram_bfs(
            oram, source, budget, dummy_pool(adjacency), random.Random(SEED)
        )
        reference = bfs(adjacency, source)
        assert result.dist == reference.dist
        assert result.order == reference.order
        assert result.burst_lengths == [1 + budget] * len(adjacency)


def test_mitigated_matches_reference_on_random_graph() -> None:
    adjacency = random_graph()
    budget = max_degree(adjacency)
    oram = make_oram(adjacency, debug=False)
    for source in range(0, len(adjacency), 7):
        result = mitigated_oram_bfs(
            oram, source, budget, dummy_pool(adjacency), random.Random(SEED)
        )
        reference = bfs(adjacency, source)
        assert (result.dist, result.order) == (reference.dist, reference.order)
        assert set(result.burst_lengths) == {1 + budget}
    oram.check_invariant()


@pytest.mark.parametrize(
    ("graph", "budget", "fingerprint"),
    [("chain", 2, [3, 3, 3, 3, 3]), ("star", 4, [5, 5, 5, 5, 5])],
)
def test_mitigated_fingerprint_matches_appendix_b(
    graph: str, budget: int, fingerprint: list[int], request: pytest.FixtureRequest
) -> None:
    # Spec Appendix B.1 and B.2: every source gives the same burst sequence.
    adjacency: Adjacency = request.getfixturevalue(graph)
    oram = make_oram(adjacency)
    for source in range(len(adjacency)):
        result = mitigated_oram_bfs(
            oram, source, budget, dummy_pool(adjacency), random.Random(SEED)
        )
        assert result.burst_lengths == fingerprint


def test_mitigated_access_count(star: Adjacency) -> None:
    # Spec §5.3: |V| * (1 + B) logical accesses, |V| * B - 2|E| of them dummy.
    budget = max_degree(star)
    oram = make_oram(star)
    mitigated_oram_bfs(oram, 1, budget, dummy_pool(star), random.Random(SEED))
    assert len(oram.server.events) == 2 * len(star) * (1 + budget)


class RecordingRandom(random.Random):
    # Seeded generator that also records every dummy ID it picks.
    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self.picked: list[object] = []

    def choice(self, seq):
        picked = super().choice(seq)
        self.picked.append(picked)
        return picked


def test_mitigated_dummy_accesses(star: Adjacency) -> None:
    # Spec §5.3: |V| * B - 2|E| dummy accesses, all from the pool. For the
    # star that is 5 * 4 - 8 = 12: the center pads none, each leaf pads 3.
    pool = dummy_pool(star)
    rng = RecordingRandom(SEED)
    mitigated_oram_bfs(make_oram(star), 0, 4, pool, rng)
    assert len(rng.picked) == 12
    assert set(rng.picked) <= set(pool)


def test_mitigated_rejects_budget_below_max_degree(star: Adjacency) -> None:
    # Spec §2.9: with B < max degree some neighbors would not be read.
    with pytest.raises(ValueError):
        mitigated_oram_bfs(make_oram(star), 0, 3, dummy_pool(star), random.Random(SEED))


def test_mitigated_rejects_empty_dummy_pool(star: Adjacency) -> None:
    with pytest.raises(ValueError):
        mitigated_oram_bfs(make_oram(star), 0, 4, [], random.Random(SEED))
