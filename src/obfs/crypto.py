import os
import struct
from collections.abc import Sequence

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .block_codec import POSITION_BYTES, deserialize_position, serialize_position
from .domain import Block

# Spec §4.4 and Appendix A.
NONCE_BYTES = 12
TAG_BYTES = 16

_BUCKET_INDEX = struct.Struct(">Q")


def generate_key() -> bytes:
    return AESGCM.generate_key(bit_length=256)


# Spec §4.4, procedure SIFRUJ_BUCKET. Returns nonce || ciphertext || tag.
def encrypt_bucket(bucket: Sequence[Block], bucket_index: int, key: bytes) -> bytes:
    # Spec §4.4, step 1. The caller passes exactly Z positions, with EMPTY
    # blocks in the free ones.
    plaintext = b"".join(serialize_position(block) for block in bucket)
    # Spec §4.4, step 2. The nonce comes from the OS, never from the seeded
    # experiment RNG (Appendix A, master_seed).
    nonce = os.urandom(NONCE_BYTES)
    # Spec §4.4, step 3.
    aad = _BUCKET_INDEX.pack(bucket_index)
    # Spec §4.4, steps 4-5. AESGCM appends the tag to the ciphertext.
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


# Inverse of encrypt_bucket. Raises cryptography.exceptions.InvalidTag if the
# record was changed, or was written for another bucket_index or key.
def decrypt_bucket(record: bytes, bucket_index: int, key: bytes) -> list[Block]:
    nonce, ciphertext = record[:NONCE_BYTES], record[NONCE_BYTES:]
    aad = _BUCKET_INDEX.pack(bucket_index)
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, aad)
    return [
        deserialize_position(plaintext[start : start + POSITION_BYTES])
        for start in range(0, len(plaintext), POSITION_BYTES)
    ]
