import pytest

from obfs import Block, Dummy, Header, Record, Slot
from obfs.block_codec import (
    EMPTY_BLOCK,
    EMPTY_ID,
    PAYLOAD_BYTES,
    POSITION_BYTES,
    decode_record,
    deserialize_position,
    encode_record,
    serialize_position,
)

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
