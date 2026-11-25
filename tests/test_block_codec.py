import random

import pytest

from obfs import Adjacency, Block, Dummy, Header, Record, Slot
from obfs.block_codec import (
    EMPTY_BLOCK,
    EMPTY_ID,
    EMPTY_SLOT_SENTINEL,
    PAYLOAD_BYTES,
    POSITION_BYTES,
    decode_record,
    deserialize_position,
    encode_graph,
    encode_record,
    serialize_position,
)
from obfs.path_oram import PathORAM

# Spec §2.8: blocks of the graph 0-1, 0-2, plus one dummy.
SPEC_2_8_BLOCKS: list[tuple[int, Record]] = [
    (0, Header(vertex_id=0, degree=2, first_slot_id=3)),
    (1, Header(vertex_id=1, degree=1, first_slot_id=5)),
    (2, Header(vertex_id=2, degree=1, first_slot_id=6)),
    (3, Slot(owner_id=0, neighbor_id=1)),
    (4, Slot(owner_id=0, neighbor_id=2)),
    (5, Slot(owner_id=1, neighbor_id=0)),
    (6, Slot(owner_id=2, neighbor_id=0)),
    (7, Dummy()),
]


@pytest.mark.parametrize(("block_id", "record"), SPEC_2_8_BLOCKS)
def test_record_round_trip(block_id: int, record: Record) -> None:
    payload = encode_record(record)
    assert len(payload) == PAYLOAD_BYTES
    assert decode_record(payload) == record


@pytest.mark.parametrize(("block_id", "record"), SPEC_2_8_BLOCKS)
def test_position_round_trip(block_id: int, record: Record) -> None:
    block = Block(block_id, encode_record(record))
    assert deserialize_position(serialize_position(block)) == block


def test_all_position_types_have_the_same_size() -> None:
    blocks = [
        Block(0, encode_record(Header(vertex_id=0, degree=2, first_slot_id=3))),
        Block(3, encode_record(Slot(owner_id=0, neighbor_id=1))),
        Block(7, encode_record(Dummy())),
        EMPTY_BLOCK,
    ]
    sizes = {len(serialize_position(block)) for block in blocks}
    assert sizes == {POSITION_BYTES}


def test_empty_block_has_reserved_id_and_zero_payload() -> None:
    assert EMPTY_BLOCK.block_id == EMPTY_ID
    assert EMPTY_BLOCK.payload == bytes(PAYLOAD_BYTES)
    assert deserialize_position(serialize_position(EMPTY_BLOCK)) == EMPTY_BLOCK


def test_empty_payload_is_not_a_record() -> None:
    with pytest.raises(ValueError):
        decode_record(EMPTY_BLOCK.payload)


@pytest.mark.parametrize("size", [PAYLOAD_BYTES - 1, PAYLOAD_BYTES + 1])
def test_payload_of_wrong_size_is_rejected(size: int) -> None:
    with pytest.raises(ValueError):
        decode_record(bytes(size))
    with pytest.raises(ValueError):
        serialize_position(Block(0, bytes(size)))


def test_position_of_wrong_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        deserialize_position(bytes(POSITION_BYTES - 1))


# Spec §2.8 and Appendix A: the graph 0-1, 0-2 with a pool of 64 dummies.
def test_encode_graph_matches_spec_example() -> None:
    blocks = encode_graph([[1, 2], [0], [0]], dummy_pool_size=64)
    assert [block.block_id for block in blocks] == list(range(71))
    expected = dict(SPEC_2_8_BLOCKS[:7])
    for block in blocks:
        record = decode_record(block.payload)
        assert record == expected.get(block.block_id, Dummy())


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_encode_graph_id_ranges(graph: str, request: pytest.FixtureRequest) -> None:
    # Spec §2.7: N = |V| + 2|E| + D, headers first, then slots, then dummies.
    adjacency: Adjacency = request.getfixturevalue(graph)
    n = len(adjacency)
    num_slots = sum(len(neighbors) for neighbors in adjacency)
    blocks = encode_graph(adjacency, dummy_pool_size=8)
    records = [decode_record(block.payload) for block in blocks]

    assert [block.block_id for block in blocks] == list(range(n + num_slots + 8))
    assert all(isinstance(record, Header) for record in records[:n])
    assert all(isinstance(record, Slot) for record in records[n : n + num_slots])
    assert all(isinstance(record, Dummy) for record in records[n + num_slots :])


@pytest.mark.parametrize("graph", ["chain", "star", "cycle"])
def test_header_points_to_its_slots(graph: str, request: pytest.FixtureRequest) -> None:
    # Slot first_slot_id + i holds the i-th neighbor of the header's vertex
    adjacency: Adjacency = request.getfixturevalue(graph)
    records = [decode_record(block.payload) for block in encode_graph(adjacency, 0)]
    for u, neighbors in enumerate(adjacency):
        header = records[u]
        assert isinstance(header, Header)
        assert (header.vertex_id, header.degree) == (u, len(neighbors))
        for i, v in enumerate(neighbors):
            assert records[header.first_slot_id + i] == Slot(u, v)


def test_vertex_without_neighbors_gets_sentinel() -> None:
    blocks = encode_graph([[1], [0], []], dummy_pool_size=0)
    assert decode_record(blocks[2].payload) == Header(2, 0, EMPTY_SLOT_SENTINEL)


def test_negative_dummy_pool_is_rejected() -> None:
    with pytest.raises(ValueError):
        encode_graph([[1], [0]], dummy_pool_size=-1)


def test_encoded_graph_reads_back_through_path_oram(star: Adjacency) -> None:
    oram = PathORAM(encode_graph(star, dummy_pool_size=4), random.Random(2027))
    adjacency: Adjacency = []
    for u in range(len(star)):
        header = decode_record(oram.access("read", u))
        assert isinstance(header, Header)
        neighbors = []
        for i in range(header.degree):
            slot = decode_record(oram.access("read", header.first_slot_id + i))
            assert isinstance(slot, Slot)
            neighbors.append(slot.neighbor_id)
        adjacency.append(neighbors)
    assert adjacency == star
