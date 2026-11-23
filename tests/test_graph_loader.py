from pathlib import Path

import pytest

from obfs import GraphStats
from obfs.graph_loader import load_graph


def test_chain(graphs_dir: Path) -> None:
    # Spec Appendix B.1: chain 1-2-3-4-5, remapped to 0..4.
    adjacency, stats = load_graph(graphs_dir / "chain.txt", "facebook")
    assert adjacency == [[1], [0, 2], [1, 3], [2, 4], [3]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=4, avg_degree=1.6, max_degree=2
    )


def test_star(graphs_dir: Path) -> None:
    # Spec Appendix B.2: center 0, leaves 1-4.
    adjacency, stats = load_graph(graphs_dir / "star.txt", "facebook")
    assert adjacency == [[1, 2, 3, 4], [0], [0], [0], [0]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=4, avg_degree=1.6, max_degree=4
    )


def test_cycle(graphs_dir: Path) -> None:
    # Spec Appendix B.3: five vertices.
    adjacency, stats = load_graph(graphs_dir / "cycle.txt", "facebook")
    assert adjacency == [[1, 4], [0, 2], [1, 3], [2, 4], [0, 3]]
    assert stats == GraphStats(
        num_vertices=5, num_edges=5, avg_degree=2.0, max_degree=2
    )


def test_self_loops_duplicates_and_reversed_edges_are_removed(graphs_dir: Path) -> None:
    adjacency, stats = load_graph(
        graphs_dir / "duplicates_and_self_loops.txt", "facebook"
    )
    assert adjacency == [[1], [0, 2], [1]]
    assert stats.num_edges == 2


def test_only_largest_component_is_kept(graphs_dir: Path) -> None:
    adjacency, stats = load_graph(graphs_dir / "two_components.txt", "facebook")
    assert adjacency == [[1], [0, 2], [1]]
    assert stats.num_vertices == 3


def test_ids_are_remapped_in_sorted_order(graphs_dir: Path) -> None:
    # Sorted original IDs 7, 42, 500 become 0, 1, 2.
    adjacency, _ = load_graph(graphs_dir / "unsorted_ids.txt", "facebook")
    assert adjacency == [[1, 2], [0], [0]]


def test_bitcoin_alpha_ignores_rating_and_time(graphs_dir: Path) -> None:
    # Spec §4.2, Bitcoin Alpha: both directions and repeated ratings give one edge.
    adjacency, _ = load_graph(graphs_dir / "bitcoin_alpha_sample.csv", "bitcoin_alpha")
    assert adjacency == [[1], [0, 2], [1]]


def test_file_without_edges_is_rejected(graphs_dir: Path) -> None:
    with pytest.raises(ValueError, match="no edges"):
        load_graph(graphs_dir / "self_loop_only.txt", "facebook")


def test_unknown_dataset_is_rejected(graphs_dir: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_graph(graphs_dir / "chain.txt", "imdb")  # ty: ignore[invalid-argument-type]
