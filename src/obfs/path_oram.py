import random
from collections import Counter
from collections.abc import Sequence
from typing import Literal

from .block_codec import EMPTY_BLOCK, EMPTY_ID, PAYLOAD_BYTES
from .crypto import decrypt_bucket, encrypt_bucket, generate_key
from .domain import Block, BlockId

# Leaf of the tree a block is assigned to, 0..num_leaves-1 (Spec §2.3).
type Leaf = int

# Tree node: exactly Z positions, each a real block or EMPTY (Spec §4.5).
type Bucket = list[Block]

type Operation = Literal["read", "write"]

# What the server observes on each call: the kind of event and the bucket
# indices of the path (Spec §2.6, §3.1).
type PathEvent = tuple[Literal["read_path", "write_path"], tuple[int, ...]]


class InvariantError(Exception):
    pass


class StashOverflowError(Exception):
    pass


# Spec §4.6, function PUTANJA_DO_LISTA. Buckets are numbered in heap order:
# the root is 0 and the children of node i are 2i + 1 and 2i + 2.
def path_to_leaf(leaf: Leaf, num_leaves: int) -> list[int]:
    if not 0 <= leaf < num_leaves:
        raise ValueError(f"leaf {leaf} is not in 0..{num_leaves - 1}")
    node = (num_leaves - 1) + leaf
    path = [node]
    while node != 0:
        node = (node - 1) // 2
        path.insert(0, node)
    return path


# Spec §4.5, ServerTree. The untrusted server: it stores encrypted bucket
# records and logs which paths are read and written.
class ServerTree:
    def __init__(self, records: list[bytes]) -> None:
        self.records = records
        self.events: list[PathEvent] = []

    def read_path(self, path: list[int]) -> list[bytes]:
        self.events.append(("read_path", tuple(path)))
        return [self.records[index] for index in path]

    def write_path(self, path: list[int], records: list[bytes]) -> None:
        self.events.append(("write_path", tuple(path)))
        for index, record in zip(path, records, strict=True):
            self.records[index] = record


# Spec §4.5, PathORAM. The trusted client: it holds the key, position map and
# stash, and talks to the server only through read_path and write_path.
class PathORAM:
    # Spec §4.8, algorithm INICIJALIZUJ_ORAM. Defaults follow Appendix A.
    def __init__(
        self,
        blocks: Sequence[Block],
        rng: random.Random,
        z: int = 4,
        stash_limit: int = 256,
        debug: bool = False,
    ) -> None:
        ids = [block.block_id for block in blocks]
        if not ids:
            raise ValueError("ORAM needs at least one block")
        if len(set(ids)) != len(ids) or EMPTY_ID in ids:
            raise ValueError("block IDs must be unique and not EMPTY_ID")

        self.rng = rng
        self.z = z
        self.stash_limit = stash_limit
        self.debug = debug
        self.key = generate_key()

        # Spec §4.5: the smallest power of two >= N leaves, depth L.
        self.num_leaves = 1 << (len(blocks) - 1).bit_length()
        self.depth = self.num_leaves.bit_length() - 1

        # Spec §4.8, steps 1-2.
        num_buckets = 2 * self.num_leaves - 1
        self.server = ServerTree(
            [
                encrypt_bucket([EMPTY_BLOCK] * z, index, self.key)
                for index in range(num_buckets)
            ]
        )
        self.position_map: dict[BlockId, Leaf] = {}
        self.stash: dict[BlockId, Block] = {}
        self.stash_sizes: list[int] = []

        # Spec §4.8, step 3.
        for block in sorted(blocks, key=lambda block: block.block_id):
            leaf = self._random_leaf()
            self.position_map[block.block_id] = leaf
            self.stash[block.block_id] = block
            path = path_to_leaf(leaf, self.num_leaves)
            self._read_path_into_stash(path)
            self._write_path(path)

        # Spec §4.8, steps 4-5. Only the metrics and the server log are reset,
        # the stash is kept.
        self.check_invariant()
        self.stash_sizes = []
        self.server.events = []

    # Spec §4.9, algorithm ORAM_ACCESS.
    def access(
        self, op: Operation, block_id: BlockId, new_payload: bytes | None = None
    ) -> bytes:
        # Spec §4.9, step 1. The payload is also checked here, before any
        # state changes.
        if block_id not in self.position_map:
            raise ValueError(f"block {block_id} is not stored in this ORAM")
        if op == "write" and (new_payload is None or len(new_payload) != PAYLOAD_BYTES):
            raise ValueError(f"write needs a payload of {PAYLOAD_BYTES} bytes")

        # Spec §4.9, steps 2-4.
        old_leaf = self.position_map[block_id]
        path = path_to_leaf(old_leaf, self.num_leaves)
        self._read_path_into_stash(path)

        # Spec §4.9, steps 5-7.
        if block_id not in self.stash:
            raise InvariantError(f"block {block_id} is not on its path or in the stash")
        if op == "write" and new_payload is not None:
            self.stash[block_id] = Block(block_id, new_payload)
        payload = self.stash[block_id].payload

        # Spec §4.9, steps 8-10.
        self.position_map[block_id] = self._random_leaf()
        self._write_path(path)

        # Spec §4.9, steps 11-12.
        if self.debug:
            self.check_invariant()

        # Spec §4.9, step 13.
        return payload

    # Spec §4.10, algorithm PROVERI_INVARIJANTU. Reads the whole tree
    # directly, so it is not part of what the server observes.
    def check_invariant(self) -> None:
        # Spec §4.10, steps 1-2.
        counts: Counter[BlockId] = Counter(self.stash.keys())

        # Spec §4.10, step 3.
        for index, record in enumerate(self.server.records):
            bucket = decrypt_bucket(record, index, self.key)
            if len(bucket) != self.z:
                raise InvariantError(f"bucket {index} has {len(bucket)} positions")
            for block in bucket:
                if block.block_id == EMPTY_ID:
                    continue
                counts[block.block_id] += 1
                if not self._compatible(block.block_id, index):
                    raise InvariantError(
                        f"block {block.block_id} in bucket {index} is not on its path"
                    )

        # Spec §4.10, step 4.
        for block_id in self.position_map:
            if counts[block_id] != 1:
                raise InvariantError(
                    f"block {block_id} appears {counts[block_id]} times"
                )
        unknown = counts.keys() - self.position_map.keys()
        if unknown:
            raise InvariantError(f"unregistered blocks {sorted(unknown)}")

    def _random_leaf(self) -> Leaf:
        return self.rng.randrange(self.num_leaves)

    # Spec §4.6, function KOMPATIBILAN.
    def _compatible(self, block_id: BlockId, bucket_index: int) -> bool:
        return bucket_index in path_to_leaf(
            self.position_map[block_id], self.num_leaves
        )

    # Spec §4.9, steps 3-4: read and decrypt the path, move every real block
    # into the stash. The path is left empty.
    def _read_path_into_stash(self, path: list[int]) -> None:
        records = self.server.read_path(path)
        for index, record in zip(path, records, strict=True):
            for block in decrypt_bucket(record, index, self.key):
                if block.block_id == EMPTY_ID:
                    continue
                if block.block_id in self.stash:
                    raise InvariantError(f"block {block.block_id} is duplicated")
                self.stash[block.block_id] = block

    # Spec §4.7 (procedure VRATI_PUTANJU) and §4.9, steps 9-11.
    def _write_path(self, path: list[int]) -> None:
        buckets: list[Bucket] = []
        # Spec §4.7, step 1: from the leaf (level L) up to the root.
        for bucket_index in reversed(path):
            candidates = sorted(
                block_id
                for block_id in self.stash
                if self._compatible(block_id, bucket_index)
            )[: self.z]
            bucket = [self.stash.pop(block_id) for block_id in candidates]
            bucket += [EMPTY_BLOCK] * (self.z - len(bucket))
            buckets.insert(0, bucket)

        # Spec §4.9, step 10: every bucket gets a fresh nonce.
        records = [
            encrypt_bucket(bucket, index, self.key)
            for index, bucket in zip(path, buckets, strict=True)
        ]
        self.server.write_path(path, records)

        # Spec §4.9, step 11.
        self.stash_sizes.append(len(self.stash))
        if len(self.stash) > self.stash_limit:
            raise StashOverflowError(
                f"stash has {len(self.stash)} blocks, limit is {self.stash_limit}"
            )
