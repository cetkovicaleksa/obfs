from pathlib import Path
from typing import Literal

from .domain import Adjacency, GraphStats
from .graph_loader import load_graph
from .resources import GRAPHS

# SNAP datasets the loader can read (Spec §5.1).
type SnapGraph = Literal["bitcoin_alpha", "facebook"]

# Example graphs shipped with the package (Spec Appendix B).
type ExampleGraph = Literal["chain", "star", "cycle"]


def example(name: ExampleGraph) -> tuple[Adjacency, GraphStats]:
    return load_graph(GRAPHS / f"{name}.txt", "facebook")


SNAP_URL = "https://snap.stanford.edu/data"


def snap(dataset: SnapGraph) -> tuple[Adjacency, GraphStats]:
    try:
        import pooch
    except ImportError as error:
        raise ImportError(
            "downloading SNAP datasets needs pooch, install obfs[datasets]"
        ) from error

    match dataset:
        case "bitcoin_alpha":
            name = "soc-sign-bitcoinalpha.csv.gz"
            sha256 = "3a178611b9c2f39c9a0dc75936f28557317f1733319b49da4838232b3757cd76"
        case "facebook":
            name = "facebook_combined.txt.gz"
            sha256 = "125e84db872eeba443d270c70315c256b0af43a502fcfe51f50621166ad035d7"
        case _:
            raise ValueError(f"unknown dataset {dataset!r}")

    path = pooch.retrieve(
        f"{SNAP_URL}/{name}",
        known_hash=f"sha256:{sha256}",
        fname=name,
        path=pooch.os_cache("obfs"),
    )
    return load_graph(Path(path), dataset)
