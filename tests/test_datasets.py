from obfs import GraphStats
from obfs.datasets import example


def test_example_chain() -> None:
    # Spec Appendix B.1: chain 1-2-3-4-5, remapped to 0..4.
    adjacency, stats = example("chain")
    assert adjacency == [[1], [0, 2], [1, 3], [2, 4], [3]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=4, avg_degree=1.6, max_degree=2
    )


def test_example_star() -> None:
    # Spec Appendix B.2: center 0, leaves 1-4.
    adjacency, stats = example("star")
    assert adjacency == [[1, 2, 3, 4], [0], [0], [0], [0]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=4, avg_degree=1.6, max_degree=4
    )


def test_example_cycle() -> None:
    # Spec Appendix B.3: five vertices.
    adjacency, stats = example("cycle")
    assert adjacency == [[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=5, avg_degree=2.0, max_degree=2
    )
