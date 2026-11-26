from typing import Literal

from .domain import Adjacency, GraphStats
from .graph_loader import load_graph
from .resources import GRAPHS

# Example graphs shipped with the package (Spec Appendix B).
type ExampleGraph = Literal["chain", "star", "cycle"]


# Spec Appendix B: chain 1-2-3-4-5, star with center 0, cycle of five. They
# are plain edge lists, the same format as the Facebook file.
def example(name: ExampleGraph) -> tuple[Adjacency, GraphStats]:
    return load_graph(GRAPHS / f"{name}.txt", "facebook")
