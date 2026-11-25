from .bfs_plain import bfs
from .domain import Adjacency, BfsResult, Vertex


# Spec §4.12, algorithm PLAINTEXT_BFS
def plaintext_bfs(adjacency: Adjacency, source: Vertex) -> BfsResult:
    return bfs(adjacency, source)
