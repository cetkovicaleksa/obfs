import pytest
from cryptography.exceptions import InvalidTag

from obfs import Block, Dummy, Header, Slot
from obfs.block_codec import EMPTY_BLOCK, POSITION_BYTES, encode_record
from obfs.crypto import (
    NONCE_BYTES,
    TAG_BYTES,
    decrypt_bucket,
    encrypt_bucket,
    generate_key,
)

# Appendix A.
Z = 4


@pytest.fixture
def key() -> bytes:
    return generate_key()


@pytest.fixture
def bucket() -> list[Block]:
    # Z positions: three real blocks and one free position.
    return [
        Block(0, encode_record(Header(vertex_id=0, degree=2, first_slot_id=3))),
        Block(3, encode_record(Slot(owner_id=0, neighbor_id=1))),
        Block(7, encode_record(Dummy())),
        EMPTY_BLOCK,
    ]


def test_round_trip(bucket: list[Block], key: bytes) -> None:
    record = encrypt_bucket(bucket, 5, key)
    assert decrypt_bucket(record, 5, key) == bucket


def test_empty_bucket_round_trip(key: bytes) -> None:
    bucket = [EMPTY_BLOCK] * Z
    assert decrypt_bucket(encrypt_bucket(bucket, 0, key), 0, key) == bucket


def test_record_size_matches_cost_formula(bucket: list[Block], key: bytes) -> None:
    # Spec §5.6: bucket_record_bytes = nonce + Z * position + tag.
    record = encrypt_bucket(bucket, 0, key)
    assert len(record) == NONCE_BYTES + Z * POSITION_BYTES + TAG_BYTES


def test_same_bucket_encrypts_differently(bucket: list[Block], key: bytes) -> None:
    # Spec §4.4: a fresh nonce on every write gives a different record.
    first = encrypt_bucket(bucket, 0, key)
    second = encrypt_bucket(bucket, 0, key)
    assert first[:NONCE_BYTES] != second[:NONCE_BYTES]
    assert first != second
    assert decrypt_bucket(first, 0, key) == decrypt_bucket(second, 0, key)


def test_record_moved_to_another_bucket_is_rejected(
    bucket: list[Block], key: bytes
) -> None:
    # The bucket index is the AAD (Spec §4.4, step 3).
    record = encrypt_bucket(bucket, 1, key)
    with pytest.raises(InvalidTag):
        decrypt_bucket(record, 2, key)


def test_modified_record_is_rejected(bucket: list[Block], key: bytes) -> None:
    record = bytearray(encrypt_bucket(bucket, 0, key))
    record[NONCE_BYTES] ^= 1
    with pytest.raises(InvalidTag):
        decrypt_bucket(bytes(record), 0, key)


def test_wrong_key_is_rejected(bucket: list[Block], key: bytes) -> None:
    record = encrypt_bucket(bucket, 0, key)
    with pytest.raises(InvalidTag):
        decrypt_bucket(record, 0, generate_key())
