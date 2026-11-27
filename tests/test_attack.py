import random
from dataclasses import astuple

import pytest

from obfs import Adjacency, AttackMetrics, AttackResult
from obfs.attack import (
    attack_query,
    build_fingerprint_database,
    compute_attack_metrics,
    select_test_sources,
)
from obfs.bfs_engines import mitigated_oram_bfs, naive_oram_bfs
from obfs.bfs_plain import bfs
from obfs.block_codec import encode_graph
from obfs.path_oram import PathORAM

# Appendix B with vertices loaded as 0..n-1: chain 1-2-3-4-5 is 0..4, the
# star center is 0.


def test_chain_naive_database(chain: Adjacency) -> None:
    # Spec Appendix B.1.
    assert build_fingerprint_database(chain) == {
        (2, 3, 3, 3, 2): [0, 4],
        (3, 2, 3, 3, 2): [1],
        (3, 3, 3, 2, 2): [2],
        (3, 3, 2, 3, 2): [3],
    }


def test_star_naive_database(star: Adjacency) -> None:
    # Spec Appendix B.2.
    assert build_fingerprint_database(star) == {
        (5, 2, 2, 2, 2): [0],
        (2, 5, 2, 2, 2): [1, 2, 3, 4],
    }


def test_cycle_naive_database(cycle: Adjacency) -> None:
    # Spec Appendix B.3.
    assert build_fingerprint_database(cycle) == {(3, 3, 3, 3, 3): [0, 1, 2, 3, 4]}


@pytest.mark.parametrize(
    ("graph", "budget", "fingerprint"),
    [("chain", 2, (3, 3, 3, 3, 3)), ("star", 4, (5, 5, 5, 5, 5))],
)
def test_mitigated_database(
    graph: str,
    budget: int,
    fingerprint: tuple[int, ...],
    request: pytest.FixtureRequest,
) -> None:
    # Spec Appendix B.1 and B.2, mitigated fingerprints.
    adjacency: Adjacency = request.getfixturevalue(graph)
    assert build_fingerprint_database(adjacency, budget) == {
        fingerprint: [0, 1, 2, 3, 4]
    }


def test_attack_query_on_chain(chain: Adjacency) -> None:
    database = build_fingerprint_database(chain)
    assert attack_query([2, 3, 3, 3, 2], database) == AttackResult([0, 4], 2, 0.5)
    assert attack_query([3, 2, 3, 3, 2], database) == AttackResult([1], 1, 1.0)


@pytest.mark.parametrize(
    ("graph", "tie_sizes"),
    [
        ("chain", [2, 1, 1, 1, 2]),
        ("star", [1, 4, 4, 4, 4]),
        ("cycle", [5, 5, 5, 5, 5]),
    ],
)
def test_attack_finds_source_in_tie_group(
    graph: str, tie_sizes: list[int], request: pytest.FixtureRequest
) -> None:
    # Spec §3.3, step 4: the true source is always in its tie group.
    adjacency: Adjacency = request.getfixturevalue(graph)
    database = build_fingerprint_database(adjacency)
    for source, tie_size in enumerate(tie_sizes):
        result = attack_query(bfs(adjacency, source).burst_lengths, database)
        assert source in result.candidates
        assert result.tie_size == tie_size


def test_attack_query_rejects_unknown_fingerprint(chain: Adjacency) -> None:
    with pytest.raises(KeyError):
        attack_query([9, 9], build_fingerprint_database(chain))


@pytest.mark.parametrize(
    ("graph", "metrics"),
    [
        ("chain", AttackMetrics(0.8, 1.4, 0.6)),
        ("star", AttackMetrics(0.4, 3.4, 0.2)),
        ("cycle", AttackMetrics(0.2, 5.0, 0.0)),
    ],
)
def test_naive_metrics_over_all_sources(
    graph: str, metrics: AttackMetrics, request: pytest.FixtureRequest
) -> None:
    adjacency: Adjacency = request.getfixturevalue(graph)
    database = build_fingerprint_database(adjacency)
    results = [
        attack_query(bfs(adjacency, s).burst_lengths, database)
        for s in range(len(adjacency))
    ]
    assert astuple(compute_attack_metrics(results, database)) == pytest.approx(
        astuple(metrics)
    )


def test_attack_on_engine_output(star: Adjacency) -> None:
    # Spec §3.5: naive leaks the star center, mitigated gives 1/|C|. The
    # attack sees only burst_lengths (Spec §6.3).
    budget = 4
    blocks = encode_graph(star, 4)
    dummy_pool = [block.block_id for block in blocks[-4:]]
    rng = random.Random(2027)
    naive_db = build_fingerprint_database(star)
    mitigated_db = build_fingerprint_database(star, budget)

    naive, mitigated = [], []
    for s in range(len(star)):
        oram = PathORAM(blocks, rng)
        naive.append(attack_query(naive_oram_bfs(oram, s).burst_lengths, naive_db))
        oram = PathORAM(blocks, rng)
        observed = mitigated_oram_bfs(oram, s, budget, dummy_pool, rng).burst_lengths
        mitigated.append(attack_query(observed, mitigated_db))

    assert naive[0] == AttackResult([0], 1, 1.0)
    assert astuple(compute_attack_metrics(mitigated, mitigated_db)) == pytest.approx(
        (1 / len(star), len(star), 0.0)
    )


@pytest.fixture
def long_chain() -> Adjacency:
    # 200 vertices: degree 1 at the ends, 2 elsewhere.
    n = 200
    return [[v for v in (u - 1, u + 1) if 0 <= v < n] for u in range(n)]


def test_test_sources_are_ten_per_degree_group(long_chain: Adjacency) -> None:
    # Spec §5.2: ranked by (degree, id) is 0, 199, 1, ..., 198, in ten
    # groups of 20.
    sources = select_test_sources(long_chain, random.Random(2027))
    ranked = [0, 199, *range(1, 199)]
    assert len(set(sources)) == 100
    for i in range(10):
        assert sum(1 for s in sources if s in ranked[20 * i : 20 * (i + 1)]) == 10


def test_test_sources_follow_count(long_chain: Adjacency) -> None:
    assert len(select_test_sources(long_chain, random.Random(1), count=20)) == 20


def test_test_sources_are_reproducible(long_chain: Adjacency) -> None:
    first = select_test_sources(long_chain, random.Random(2027))
    assert select_test_sources(long_chain, random.Random(2027)) == first


def test_small_graph_uses_every_vertex(chain: Adjacency) -> None:
    # Spec §5.2, step 4: five vertices give five groups of one.
    assert select_test_sources(chain, random.Random(0)) == [0, 4, 1, 2, 3]


def test_more_than_100_sources_are_rejected(chain: Adjacency) -> None:
    with pytest.raises(ValueError, match="0..100"):
        select_test_sources(chain, random.Random(0), count=101)
