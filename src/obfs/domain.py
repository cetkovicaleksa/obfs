from dataclasses import dataclass
from typing import Literal, Self

# Vertex ID. In a dataset file it is the original ID; after loading it is
# remapped to 0..n-1 (Spec §4.2, step 5).
type Vertex = int

# Undirected edge {u, v}.
type Edge = tuple[Vertex, Vertex]

# adjacency[u] is the sorted list of neighbors of vertex u (Spec §4.2, step 6).
type Adjacency = list[list[Vertex]]

# Number of logical accesses while BFS processes one vertex: one header read
# plus one read per neighbor slot, 1 + deg(u) (Spec §2.1).
type Burst = int

# Datasets the loader can read (Spec §5.1).
type Dataset = Literal["bitcoin_alpha", "facebook"]


@dataclass(frozen=True)
class GraphStats:
    num_vertices: int
    num_edges: int
    avg_degree: float
    max_degree: int

    @classmethod
    def from_adjacency(cls, adjacency: Adjacency) -> Self:
        # Spec §4.2, step 7.
        degrees = [len(neighbors) for neighbors in adjacency]
        return cls(
            num_vertices=len(adjacency),
            num_edges=sum(degrees) // 2,
            avg_degree=sum(degrees) / len(adjacency),
            max_degree=max(degrees),
        )


@dataclass(frozen=True)
class BfsResult:
    # Spec §4.3 output.
    dist: dict[Vertex, int]
    order: list[Vertex]
    burst_lengths: list[Burst]


# ID of a logical ORAM block. HEADER blocks use the vertex ID, SLOT and DUMMY
# blocks follow them (Spec §2.8).
type BlockId = int


# Spec §2.7, HEADER: describes vertex_id and where its slots start.
@dataclass(frozen=True)
class Header:
    vertex_id: Vertex
    degree: int
    first_slot_id: BlockId


# Spec §2.7, SLOT: one directed record owner -> neighbor.
@dataclass(frozen=True)
class Slot:
    owner_id: Vertex
    neighbor_id: Vertex


# Spec §2.7, DUMMY: only adds an access, carries nothing BFS uses.
@dataclass(frozen=True)
class Dummy:
    pass


# Content of a logical block, before it is encoded into a payload.
type Record = Header | Slot | Dummy


# One position in a bucket: a block ID and a fixed-size payload (Spec §4.4).
@dataclass(frozen=True)
class Block:
    block_id: BlockId
    payload: bytes
