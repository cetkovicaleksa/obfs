import struct
from enum import IntEnum

from .domain import Adjacency, Block, Dummy, Header, Record, Slot

# Spec §4.4 and Appendix A.
BLOCK_ID_BYTES = 8
PAYLOAD_BYTES = 64
POSITION_BYTES = BLOCK_ID_BYTES + PAYLOAD_BYTES

# Spec §2.7, EMPTY: a reserved block_id with a zero payload. The largest
# 8-byte ID is never reached by real blocks (|V| + 2|E| + D is far smaller).
EMPTY_ID = 2 ** (8 * BLOCK_ID_BYTES) - 1
EMPTY_BLOCK = Block(EMPTY_ID, bytes(PAYLOAD_BYTES))

# Spec §4.11: first_slot_id of a vertex without neighbors. It is never a
# real block ID.
EMPTY_SLOT_SENTINEL = EMPTY_ID

_BLOCK_ID = struct.Struct(">Q")
# The first payload byte is the record type (Spec §2.7). Integer fields use
# the same 8 bytes as block IDs. The rest of the payload is zero padding.
_TYPE = struct.Struct(">B")
_HEADER = struct.Struct(">BQQQ")
_SLOT = struct.Struct(">BQQ")


class _RecordType(IntEnum):
    # 0 is left out, so an EMPTY payload is never mistaken for a record.
    HEADER = 1
    SLOT = 2
    DUMMY = 3


# Spec §2.7: HEADER, SLOT and DUMMY payloads all have PAYLOAD_BYTES.
def encode_record(record: Record) -> bytes:
    match record:
        case Header(vertex_id, degree, first_slot_id):
            fields = _HEADER.pack(_RecordType.HEADER, vertex_id, degree, first_slot_id)
        case Slot(owner_id, neighbor_id):
            fields = _SLOT.pack(_RecordType.SLOT, owner_id, neighbor_id)
        case Dummy():
            fields = _TYPE.pack(_RecordType.DUMMY)
    return fields.ljust(PAYLOAD_BYTES, b"\x00")


def decode_record(payload: bytes) -> Record:
    _check_payload(payload)
    match payload[0]:
        case _RecordType.HEADER:
            _, vertex_id, degree, first_slot_id = _HEADER.unpack_from(payload)
            return Header(vertex_id, degree, first_slot_id)
        case _RecordType.SLOT:
            _, owner_id, neighbor_id = _SLOT.unpack_from(payload)
            return Slot(owner_id, neighbor_id)
        case _RecordType.DUMMY:
            return Dummy()
        case record_type:
            raise ValueError(f"unknown record type {record_type}")


# Spec §4.4: a bucket position is block_id followed by the payload.
def serialize_position(block: Block) -> bytes:
    _check_payload(block.payload)
    return _BLOCK_ID.pack(block.block_id) + block.payload


def deserialize_position(data: bytes) -> Block:
    if len(data) != POSITION_BYTES:
        raise ValueError(f"position has {len(data)} bytes, expected {POSITION_BYTES}")
    (block_id,) = _BLOCK_ID.unpack_from(data)
    return Block(block_id, data[BLOCK_ID_BYTES:])


def _check_payload(payload: bytes) -> None:
    if len(payload) != PAYLOAD_BYTES:
        raise ValueError(f"payload has {len(payload)} bytes, expected {PAYLOAD_BYTES}")


# Spec §4.11, algorithm KODIRAJ_GRAF. Block IDs: headers 0..n-1, then the
# slots of each vertex in adjacency order, then the dummy pool.
def encode_graph(adjacency: Adjacency, dummy_pool_size: int) -> list[Block]:
    if dummy_pool_size < 0:
        raise ValueError(f"dummy pool size {dummy_pool_size} is negative")

    # Spec §4.11, steps 1-3.
    n = len(adjacency)
    headers: list[Block] = []
    slots: list[Block] = []
    slot_cursor = n

    # Spec §4.11, step 4.
    for u, neighbors in enumerate(adjacency):
        degree = len(neighbors)
        first_slot_id = slot_cursor if degree > 0 else EMPTY_SLOT_SENTINEL
        headers.append(Block(u, encode_record(Header(u, degree, first_slot_id))))
        for v in neighbors:
            slots.append(Block(slot_cursor, encode_record(Slot(u, v))))
            slot_cursor += 1

    # Spec §4.11, step 5. Every dummy has the same payload.
    # https://www.youtube.com/watch?v=nAgSecmB9lM&t=13s
    dummy = encode_record(Dummy())
    dummies = [Block(slot_cursor + j, dummy) for j in range(dummy_pool_size)]

    # Spec §4.11, step 6. The three lists are already in block_id order.
    return headers + slots + dummies
