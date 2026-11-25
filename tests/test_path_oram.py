import random

import pytest

from obfs import Block
from obfs.block_codec import EMPTY_BLOCK, PAYLOAD_BYTES
from obfs.crypto import decrypt_bucket, encrypt_bucket
from obfs.path_oram import (
    InvariantError,
    PathORAM,
    StashOverflowError,
    path_to_leaf,
)

SEED = 2027


def payload(value: int) -> bytes:
    return value.to_bytes(PAYLOAD_BYTES, "big")


def make_blocks(n: int) -> list[Block]:
    return [Block(block_id, payload(block_id)) for block_id in range(n)]


def bucket_ids(oram: PathORAM, index: int) -> list[int]:
    bucket = decrypt_bucket(oram.server.records[index], index, oram.key)
    return [block.block_id for block in bucket if block != EMPTY_BLOCK]


def block_in_tree(oram: PathORAM) -> tuple[int, int]:
    # A block stored below the root, and the index of its bucket.
    for index in range(1, len(oram.server.records)):
        ids = bucket_ids(oram, index)
        if ids:
            return ids[0], index
    raise AssertionError("every block is in the root or the stash")


class ScriptedRandom(random.Random):
    # Returns the given leaves in order, so a test can fix the remapping.
    def __init__(self, leaves: list[int]) -> None:
        super().__init__()
        self.leaves = leaves

    def randrange(self, *args: object, **kwargs: object) -> int:
        return self.leaves.pop(0)


# Spec Appendix B.4: R = 0, A = 1, B = 2, L0..L3 = 3..6.
@pytest.mark.parametrize(
    ("leaf", "path"),
    [(0, [0, 1, 3]), (1, [0, 1, 4]), (2, [0, 2, 5]), (3, [0, 2, 6])],
)
def test_paths_in_four_leaf_tree(leaf: int, path: list[int]) -> None:
    assert path_to_leaf(leaf, 4) == path


@pytest.mark.parametrize("leaf", [-1, 4])
def test_invalid_leaf_is_rejected(leaf: int) -> None:
    with pytest.raises(ValueError):
        path_to_leaf(leaf, 4)


# Spec §4.5: the smallest power of two >= N leaves and 2 * num_leaves - 1
# buckets.
@pytest.mark.parametrize(
    ("n", "num_leaves", "depth"),
    [(1, 1, 0), (2, 2, 1), (4, 4, 2), (5, 8, 3), (1024, 1024, 10)],
)
def test_tree_shape(n: int, num_leaves: int, depth: int) -> None:
    oram = PathORAM(make_blocks(n), random.Random(SEED))
    assert oram.num_leaves == num_leaves
    assert oram.depth == depth
    assert len(oram.server.records) == 2 * num_leaves - 1


def test_initialization_keeps_stash_but_resets_metrics() -> None:
    # Spec §4.8, step 5.
    oram = PathORAM(make_blocks(64), random.Random(SEED))
    assert oram.stash_sizes == []
    assert len(oram.stash) <= oram.stash_limit


def test_spec_example_two_accesses_to_b1() -> None:
    # Spec §2.5 and Appendix B.4. Blocks b1..b4 have IDs 1..4, Z = 2.
    blocks = [Block(block_id, payload(block_id)) for block_id in range(1, 5)]
    oram = PathORAM(blocks, random.Random(SEED), z=2)

    # Replace the random initial state with the one from the spec.
    oram.position_map = {1: 1, 2: 3, 3: 0, 4: 2}
    oram.stash = {}
    contents = {0: [4], 1: [3], 4: [1], 6: [2]}
    for index in range(len(oram.server.records)):
        bucket = [blocks[block_id - 1] for block_id in contents.get(index, [])]
        bucket += [EMPTY_BLOCK] * (2 - len(bucket))
        oram.server.records[index] = encrypt_bucket(bucket, index, oram.key)
    oram.check_invariant()

    # First access: read path R-A-L1, b1 moves to leaf 3.
    oram.rng = ScriptedRandom([3, 0])
    assert oram.access("read", 1) == payload(1)
    assert bucket_ids(oram, 0) == [1, 4]
    assert bucket_ids(oram, 1) == [3]
    assert bucket_ids(oram, 4) == []
    assert bucket_ids(oram, 6) == [2]
    assert oram.stash == {}

    # Second access: read path R-B-L3, b1 moves to leaf 0.
    assert oram.access("read", 1) == payload(1)
    assert bucket_ids(oram, 0) == [1]
    assert bucket_ids(oram, 2) == [4]
    assert bucket_ids(oram, 6) == [2]
    assert oram.stash == {}
    assert oram.position_map[1] == 0
    oram.check_invariant()


# Spec §6.1, C3: read/write and invariant tests on 4-1024 blocks.
@pytest.mark.parametrize("n", [4, 16, 128, 1024])
def test_reads_return_last_written_payload(n: int) -> None:
    rng = random.Random(SEED)
    oram = PathORAM(make_blocks(n), random.Random(SEED), debug=n <= 128)
    expected = {block_id: payload(block_id) for block_id in range(n)}
    for step in range(200):
        block_id = rng.randrange(n)
        if rng.random() < 0.5:
            expected[block_id] = payload(n + step)
            assert (
                oram.access("write", block_id, expected[block_id]) == expected[block_id]
            )
        else:
            assert oram.access("read", block_id) == expected[block_id]
    oram.check_invariant()
    for block_id in range(n):
        assert oram.access("read", block_id) == expected[block_id]


def test_access_remaps_block_to_new_leaf() -> None:
    # Spec §4.14: the leaf changes over repeated accesses; a single random
    # repeat is allowed.
    oram = PathORAM(make_blocks(64), random.Random(SEED))
    leaves = []
    for _ in range(20):
        oram.access("read", 7)
        leaves.append(oram.position_map[7])
    assert len(set(leaves)) > 1


def test_stash_size_is_recorded_per_access() -> None:
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    for block_id in range(10):
        oram.access("read", block_id)
    assert len(oram.stash_sizes) == 10


def test_initialization_resets_server_log() -> None:
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    assert oram.server.events == []


def test_access_is_one_read_path_and_one_write_path() -> None:
    # Spec §2.6: one logical access gives one read_path and one write_path of
    # the same path, the one to the block's leaf before the access.
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    for block_id in range(10):
        path = tuple(path_to_leaf(oram.position_map[block_id], oram.num_leaves))
        oram.server.events = []
        oram.access("read", block_id)
        assert oram.server.events == [("read_path", path), ("write_path", path)]


def test_stash_overflow_is_reported() -> None:
    # Four blocks on leaf 0 with Z = 1: its path holds only three.
    with pytest.raises(StashOverflowError):
        PathORAM(make_blocks(4), ScriptedRandom([0, 0, 0, 0]), z=1, stash_limit=0)


def test_unknown_block_is_rejected() -> None:
    oram = PathORAM(make_blocks(4), random.Random(SEED))
    with pytest.raises(ValueError):
        oram.access("read", 4)


@pytest.mark.parametrize("new_payload", [None, bytes(PAYLOAD_BYTES - 1)])
def test_write_without_valid_payload_is_rejected(new_payload: bytes | None) -> None:
    oram = PathORAM(make_blocks(4), random.Random(SEED))
    with pytest.raises(ValueError):
        oram.access("write", 0, new_payload)
    oram.check_invariant()


def test_duplicate_block_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        PathORAM([Block(0, payload(0)), Block(0, payload(1))], random.Random(SEED))


def test_invariant_detects_lost_block() -> None:
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    for index in range(len(oram.server.records)):
        oram.server.records[index] = encrypt_bucket([EMPTY_BLOCK] * 4, index, oram.key)
    oram.stash = {}
    with pytest.raises(InvariantError):
        oram.check_invariant()


def test_invariant_detects_duplicated_block() -> None:
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    block_id, _ = block_in_tree(oram)
    oram.stash[block_id] = Block(block_id, payload(block_id))
    with pytest.raises(InvariantError):
        oram.check_invariant()


def test_invariant_detects_block_off_its_path() -> None:
    oram = PathORAM(make_blocks(16), random.Random(SEED))
    block_id, index = block_in_tree(oram)
    # Assign the block a leaf whose path does not pass through its bucket.
    oram.position_map[block_id] = next(
        leaf
        for leaf in range(oram.num_leaves)
        if index not in path_to_leaf(leaf, oram.num_leaves)
    )
    with pytest.raises(InvariantError):
        oram.check_invariant()
