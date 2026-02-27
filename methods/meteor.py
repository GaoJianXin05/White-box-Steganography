# methods/meteor.py
# Meteor: Cryptographically Secure Steganography for Realistic Distributions
# Kaptchuk et al., CCS 2021
#
# Framework adapter for this project:
#   StegoMethod.step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)
#
# Notes:
# - This is a generation-only implementation (encoding). Decoding is not provided here.
# - It uses the candidate distribution already produced by ModelHandler.process_probs()
#   (temperature/top-k/top-p + candidate cap are applied there).

from __future__ import annotations

import os
import hashlib
import hmac
import numpy as np
import torch

# =========================
# Hyperparameters (env)
# =========================
# β: bit precision / max bits per step
METEOR_PRECISION = int(os.getenv("METEOR_PRECISION", "16"))

# Optional: hex-encoded key for DRBG (recommended 64 bytes -> 128 hex chars).
# If empty, we derive a key from GlobalConfig.SEED.
METEOR_KEY_HEX = os.getenv("METEOR_KEY_HEX", "")

# Optional: hex-encoded nonce/seed for DRBG.
METEOR_NONCE_HEX = os.getenv("METEOR_NONCE_HEX", "")

# Safety: require precision in [1,24] unless explicitly overridden.
METEOR_ALLOW_ANY_PRECISION = int(os.getenv("METEOR_ALLOW_ANY_PRECISION", "0"))

_PRINT_ONCE = False


def _print_once():
    global _PRINT_ONCE
    if not _PRINT_ONCE:
        print(
            f"[Meteor] METEOR_PRECISION={METEOR_PRECISION}, "
            f"KEY={'env' if METEOR_KEY_HEX else 'derived'}, "
            f"NONCE={'env' if METEOR_NONCE_HEX else 'default'}"
        )
        _PRINT_ONCE = True


# =========================
# Helpers
# =========================
def _xor_bitstrings(a: str, b: str) -> str:
    # a, b: equal-length bit strings
    return "".join("1" if (ca != cb) else "0" for ca, cb in zip(a, b))


def _common_prefix_len_msb(a: int, b: int, precision: int) -> int:
    """
    Count common MSB prefix length between a and b,
    both represented on `precision` bits.
    """
    sa = format(int(a), f"0{precision}b")
    sb = format(int(b), f"0{precision}b")
    n = 0
    for ca, cb in zip(sa, sb):
        if ca != cb:
            break
        n += 1
    return n


def _quantize_probs_to_counts(probs: np.ndarray, total: int) -> np.ndarray:
    """
    Deterministically convert probability vector to nonnegative integer counts
    that sum exactly to `total` using the largest remainder method.
    """
    if total <= 0 or probs.size == 0:
        return np.zeros_like(probs, dtype=np.int64)

    probs = np.maximum(probs.astype(np.float64, copy=False), 0.0)
    s = float(probs.sum())
    if not (s > 0.0):
        probs = np.ones_like(probs, dtype=np.float64) / float(probs.size)
    else:
        probs /= s

    raw = probs * float(total)
    base = np.floor(raw).astype(np.int64)
    frac = raw - base

    rem = int(total - int(base.sum()))
    if rem > 0:
        order = np.lexsort((np.arange(frac.size), -frac))
        base[order[:rem]] += 1
    elif rem < 0:
        need = -rem
        mask = base > 0
        if mask.any():
            frac2 = frac.copy()
            frac2[~mask] = np.inf
            order = np.lexsort((np.arange(frac2.size), frac2))
            taken = 0
            for idx in order:
                if base[idx] > 0:
                    base[idx] -= 1
                    taken += 1
                    if taken >= need:
                        break
        base = np.maximum(base, 0)

    diff = int(total - int(base.sum()))
    if diff != 0:
        base[-1] = max(0, int(base[-1] + diff))

    return base.astype(np.int64)


# =========================
# Simple HMAC-DRBG
# =========================
class _HMACDRBG:
    """
    Simple HMAC-DRBG-style generator (research-style),
    sufficient for Meteor's masking.
    """

    __slots__ = ("key", "val", "byte_index", "bit_index")

    def __init__(self, key: bytes, seed: bytes):
        if not isinstance(key, (bytes, bytearray)) or len(key) == 0:
            raise ValueError("DRBG key must be non-empty bytes.")
        self.key = bytes(key)
        self.val = b"\x01" * 64
        self.byte_index = 0
        self.bit_index = 0
        self.reseed(seed)

    @staticmethod
    def _hmac(key: bytes, data: bytes) -> bytes:
        return hmac.new(key, data, hashlib.sha512).digest()

    def reseed(self, data: bytes = b""):
        self.key = self._hmac(self.key, self.val + b"\x00" + data)
        self.val = self._hmac(self.key, self.val)
        if data:
            self.key = self._hmac(self.key, self.val + b"\x01" + data)
            self.val = self._hmac(self.key, self.val)

    def generate_bits_str(self, n: int) -> str:
        """
        Generate n bits, returned as an MSB-first bitstring.
        """
        if n <= 0:
            return ""
        out = []
        for _ in range(n):
            byte = self.val[self.byte_index]
            bit = (byte >> (7 - self.bit_index)) & 1
            out.append("1" if bit else "0")

            self.bit_index += 1
            if self.bit_index >= 8:
                self.bit_index = 0
                self.byte_index += 1

            if self.byte_index >= len(self.val):
                self.byte_index = 0
                self.val = self._hmac(self.key, self.val)

        # reseed with empty data (like the Colab demo)
        self.reseed(b"")
        return "".join(out)


# =========================
# One Meteor step
# =========================
def _meteor_step(sorted_probs: torch.Tensor,
                 sorted_indices: torch.Tensor,
                 bit_reader,
                 precision: int,
                 drbg: _HMACDRBG):
    """
    Single Meteor token decision:

      1) Peek `precision` bits from bit_reader (do NOT consume), pad with '0'.
      2) Generate DRBG mask of length `precision` and XOR -> r_bits.
      3) Interpret r_bits as integer r in [0, 2^precision).
      4) Quantize probs into integer counts summing to 2^precision.
      5) Use r to pick the token (CDF).
      6) Compute common MSB prefix length of [low, high-1] interval -> num_bits.
      7) Consume num_bits bits from bit_reader and return (token_id, consumed_bits).
    """
    if precision <= 0 or sorted_probs.numel() == 0:
        return int(sorted_indices[0].item()), ""

    total = 1 << int(precision)

    # Move to CPU float64 for stable quantization
    probs = sorted_probs.detach().to("cpu", dtype=torch.float64).numpy()
    indices = sorted_indices.detach().to("cpu").numpy()

    counts = _quantize_probs_to_counts(probs, total)
    cum = np.cumsum(counts, dtype=np.int64)
    if cum.size == 0:
        return int(indices[0]), ""
    cum[-1] = total  # ensure exact top

    # --- Peek precision bits from bit_reader (no consumption) ---
    snap = bit_reader.snapshot()
    msg_peek = bit_reader.read_bits(precision)
    bit_reader.restore(snap)

    if msg_peek is None:
        msg_peek = ""
    if len(msg_peek) < precision:
        msg_peek = msg_peek + ("0" * (precision - len(msg_peek)))
    else:
        msg_peek = msg_peek[:precision]

    # --- Mask and compute r ---
    mask_bits = drbg.generate_bits_str(precision)
    r_bits = _xor_bitstrings(msg_peek, mask_bits)
    r = int(r_bits, 2)

    # --- Use CDF to pick token ---
    pos = int(np.searchsorted(cum, r + 1, side="left"))
    if pos >= cum.size:
        pos = int(cum.size - 1)

    low = int(cum[pos - 1]) if pos > 0 else 0
    high = int(cum[pos])

    if high <= low:
        token_id = int(indices[pos])
        return token_id, ""

    # Use inclusive upper bound high-1 (as in Meteor / arithmetic coding style)
    num_bits = _common_prefix_len_msb(low, high - 1, precision)

    consumed = bit_reader.read_bits(num_bits) if num_bits > 0 else ""
    if consumed is None:
        consumed = ""

    token_id = int(indices[pos])
    return token_id, consumed


# =========================
# Framework adapter
# =========================
class StegoMethod:
    """
    Meteor method adapter.

    Required interface:
      - __init__(config, tokenizer)
      - reset_sentence()
      - step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        _print_once()

        self.precision = int(METEOR_PRECISION)
        if (self.precision < 1 or self.precision > 24) and not METEOR_ALLOW_ANY_PRECISION:
            raise ValueError(
                f"METEOR_PRECISION={self.precision} is unusual for truncated candidate sets. "
                f"Use 1~24 (default 16), or set METEOR_ALLOW_ANY_PRECISION=1 to override."
            )

        # Determine DRBG key
        if METEOR_KEY_HEX:
            key = bytes.fromhex(METEOR_KEY_HEX)
        else:
            # Derive from global SEED for repeatability
            seed_bytes = str(getattr(config, "SEED", 0)).encode("utf-8")
            key = hashlib.sha512(b"meteor-key-derive:" + seed_bytes).digest()

        # Determine nonce
        if METEOR_NONCE_HEX:
            nonce = bytes.fromhex(METEOR_NONCE_HEX)
        else:
            nonce = b"\x00" * 16

        # Initialize DRBG
        self._drbg = _HMACDRBG(key=key, seed=b"meteor-sample:" + nonce)

    def reset_sentence(self):
        """
        Meteor is stateful across tokens via DRBG.
        For *generation only* we do not reset between sentences, so the bitstream is continuous.
        If you later implement decoding with sentence-level rollback, you'll want DRBG snapshot/restore.
        """
        pass

    def step(self, sorted_probs, sorted_indices, bit_reader):
        token_id, bits = _meteor_step(
            sorted_probs=sorted_probs,
            sorted_indices=sorted_indices,
            bit_reader=bit_reader,
            precision=self.precision,
            drbg=self._drbg,
        )
        return token_id, bits
