import gzip
import sys
from pathlib import Path

import pooch
import pytest

from obfs import GraphStats
from obfs.datasets import SnapGraph, example, snap
from obfs.graph_loader import load_graph


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


@pytest.mark.parametrize(
    ("dataset", "sample", "name", "sha256"),
    [
        (
            "bitcoin_alpha",
            "bitcoin_alpha_sample.csv",
            "soc-sign-bitcoinalpha.csv.gz",
            "3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76",
        ),
        (
            "facebook",
            "unsorted_ids.txt",
            "facebook_combined.txt.gz",
            "125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7",
        ),
    ],
)
def test_snap_fetches_pinned_file(
    monkeypatch: pytest.MonkeyPatch,
    graphs_dir: Path,
    tmp_path: Path,
    dataset: SnapGraph,
    sample: str,
    name: str,
    sha256: str,
) -> None:
    # Stands in for the download: a gzipped sample in the dataset's format.
    calls: list[dict[str, object]] = []

    def retrieve(url: str, **kwargs: object) -> str:
        calls.append({"url": url, **kwargs})
        downloaded = tmp_path / name
        downloaded.write_bytes(gzip.compress((graphs_dir / sample).read_bytes()))
        return str(downloaded)

    monkeypatch.setattr(pooch, "retrieve", retrieve)
    assert snap(dataset) == load_graph(graphs_dir / sample, dataset)
    assert calls == [
        {
            "url": f"https://snap.stanford.edu/data/{name}",
            "known_hash": f"sha256:{sha256}",
            "fname": name,
            "path": pooch.os_cache("obfs"),
        }
    ]


def test_snap_without_pooch_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pooch", None)
    with pytest.raises(ImportError, match=r"obfs\[datasets\]"):
        snap("facebook")


def test_snap_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        snap("imdb")  # ty: ignore[invalid-argument-type]
