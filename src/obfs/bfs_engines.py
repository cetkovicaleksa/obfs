import random
from collections import deque
from collections.abc import Sequence

from .bfs_plain import bfs
from .block_codec import decode_record
from .domain import Adjacency, BfsResult, BlockId, Burst, Header, Slot, Vertex
from .path_oram import PathORAM


# Spec §4.12, algorithm PLAINTEXT_BFS.
def plaintext_bfs(adjacency: Adjacency, source: Vertex) -> BfsResult:
    return bfs(adjacency, source)


# Spec §4.12, algorithm NAIVE_ORAM_BFS.
def naive_oram_bfs(oram: PathORAM, source: Vertex) -> BfsResult:
    # Spec §4.12, step 1.
    dist: dict[Vertex, int] = {source: 0}
    queue = deque([source])
    order: list[Vertex] = []
    burst_lengths: list[Burst] = []

    # Spec §4.12, step 2.
    while queue:
        u = queue.popleft()
        order.append(u)
        header = _read_header(oram, u)
        burst = 1
        for i in range(header.degree):
            slot = _read_slot(oram, header.first_slot_id + i)
            burst += 1
            _visit(slot.neighbor_id, u, dist, queue)
        burst_lengths.append(burst)

    # Spec §4.12, step 3.
    return BfsResult(dist, order, burst_lengths)


# Spec §4.12, algorithm MITIGATED_ORAM_BFS.
def mitigated_oram_bfs(
    oram: PathORAM,
    source: Vertex,
    budget: int,
    dummy_pool: Sequence[BlockId],
    rng: random.Random,
) -> BfsResult:
    if not dummy_pool:
        raise ValueError("mitigated BFS needs a non-empty dummy pool")

    # Spec §4.12, step 1.
    dist: dict[Vertex, int] = {source: 0}
    queue = deque([source])
    order: list[Vertex] = []
    burst_lengths: list[Burst] = []

    # Spec §4.12, step 2.
    while queue:
        u = queue.popleft()
        order.append(u)
        header = _read_header(oram, u)
        if header.degree > budget:
            raise ValueError(
                f"vertex {u} has degree {header.degree}, above the budget {budget}"
            )
        burst = 1
        for i in range(budget):
            if i < header.degree:
                slot = _read_slot(oram, header.first_slot_id + i)
                _visit(slot.neighbor_id, u, dist, queue)
            else:
                oram.access("read", rng.choice(dummy_pool))
            burst += 1
        burst_lengths.append(burst)

    # Spec §4.12, step 3.
    return BfsResult(dist, order, burst_lengths)


# Spec §4.11: the header of vertex u has block_id u.
def _read_header(oram: PathORAM, u: Vertex) -> Header:
    match decode_record(oram.access("read", u)):
        case Header(vertex_id=vertex_id) as header if vertex_id == u:
            return header
        case _:
            raise ValueError(f"block {u} is not the header of vertex {u}")


def _read_slot(oram: PathORAM, slot_id: BlockId) -> Slot:
    match decode_record(oram.access("read", slot_id)):
        case Slot() as slot:
            return slot
        case _:
            raise ValueError(f"block {slot_id} is not a slot")


# The neighbor step of the reference BFS (Spec §4.3, step 5).
def _visit(v: Vertex, u: Vertex, dist: dict[Vertex, int], queue: deque[Vertex]) -> None:
    if v not in dist:
        dist[v] = dist[u] + 1
        queue.append(v)
