from pathlib import Path

from .domain import Adjacency, Dataset, Edge, GraphStats, Vertex


# Spec §4.2, algorithm UCITAJ_I_PRIPREMI_GRAF.
def load_graph(path: Path, dataset: Dataset) -> tuple[Adjacency, GraphStats]:
    # Spec §4.2, steps 1-2. Normalizing to (min, max) makes duplicates and
    # reversed copies of the same undirected edge collapse in the set.
    edges: set[Edge] = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            u, v = _parse_edge(line, dataset)
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
    if not edges:
        raise ValueError(f"{path} contains no edges")

    # Spec §4.2, step 3.
    neighbors: dict[Vertex, set[Vertex]] = {}
    for u, v in edges:
        neighbors.setdefault(u, set()).add(v)
        neighbors.setdefault(v, set()).add(u)

    # Spec §4.2, step 4.
    component = _largest_component(neighbors)

    # Spec §4.2, step 5.
    new_id = {original: index for index, original in enumerate(sorted(component))}

    # Spec §4.2, step 6. All neighbors of a component vertex are in the same
    # component, so this is the induced subgraph.
    adjacency: Adjacency = [[] for _ in new_id]
    for original, u in new_id.items():
        adjacency[u] = sorted(new_id[v] for v in neighbors[original])

    # Spec §4.2, steps 7-8.
    return adjacency, GraphStats.from_adjacency(adjacency)


def _parse_edge(line: str, dataset: Dataset) -> Edge:
    match dataset:
        case "bitcoin_alpha":
            # SOURCE,TARGET,RATING,TIME. Rating and time are ignored
            # (Spec §4.2, Bitcoin Alpha).
            u, v, _rating, _time = line.split(",")
        case "facebook":
            u, v = line.split()
        case _:
            raise ValueError(f"unknown dataset {dataset!r}")
    return int(u), int(v)


def _largest_component(neighbors: dict[Vertex, set[Vertex]]) -> set[Vertex]:
    # Spec §4.2, step 4 does not say which component wins a tie. Trying start
    # vertices in sorted order and replacing only on a strictly larger size
    # keeps the one with the smallest original ID.
    largest: set[Vertex] = set()
    seen: set[Vertex] = set()
    for start in sorted(neighbors):
        if start in seen:
            continue
        component = {start}
        stack = [start]
        while stack:
            for v in neighbors[stack.pop()]:
                if v not in component:
                    component.add(v)
                    stack.append(v)
        seen |= component
        if len(component) > len(largest):
            largest = component
    return largest
