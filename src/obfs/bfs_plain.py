from collections import deque

from .domain import Adjacency, BfsResult, Burst, Vertex


# Spec §4.3, algorithm BFS.
def bfs(adjacency: Adjacency, source: Vertex) -> BfsResult:
    if not 0 <= source < len(adjacency):
        raise ValueError(f"source {source} is not a vertex of the graph")

    # Spec §4.3, steps 1-4. Step 1 sets every distance to infinity; here a
    # vertex missing from dist means infinity, which is also how NetworkX
    # returns distances (Spec §4.3 compares against it).
    dist: dict[Vertex, int] = {source: 0}
    queue = deque([source])
    order: list[Vertex] = []
    burst_lengths: list[Burst] = []

    # Spec §4.3, step 5.
    while queue:
        u = queue.popleft()
        order.append(u)
        # Already visited neighbors count too: their slots are read before
        # BFS can tell they were seen (Spec §2.1).
        burst_lengths.append(1 + len(adjacency[u]))
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)

    # Spec §4.3, step 6.
    return BfsResult(dist, order, burst_lengths)
