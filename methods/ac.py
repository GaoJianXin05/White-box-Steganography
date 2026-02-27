# methods/ac.py
# Arithmetic Coding stegosampling (Ziegler et al., EMNLP 2019)
# Framework-compatible version:
#   StegoMethod.step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)

from __future__ import annotations

import os
import numpy as np
import torch


# =========================
# Hyperparameters (env)
# =========================

AC_K = int(os.getenv("AC_K", "100"))
AC_PRECISION = int(os.getenv("AC_PRECISION", "20"))

_PRINT_ONCE = False


def _print_once():
    global _PRINT_ONCE
    if not _PRINT_ONCE:
        print(f"[AC] AC_K={AC_K}, AC_PRECISION={AC_PRECISION}")
        _PRINT_ONCE = True


# =========================
# Helper: quantize probs -> integer counts summing to total
# (Deterministic; minimizes rounding error)
# =========================
def _quantize_probs_to_counts(probs: np.ndarray, total: int) -> np.ndarray:
    """
    Convert probability vector to nonnegative integer counts that sum exactly to `total`.

    We use:
      raw = probs * total
      base = floor(raw)
      distribute remainder to largest fractional parts (largest remainder method)
    This is more stable than naive rint + "add remainder to all cum boundaries".
    """
    if total <= 0:
        return np.zeros_like(probs, dtype=np.int64)

    probs = np.maximum(probs, 0.0)
    s = float(probs.sum())
    if not (s > 0.0):
        # uniform fallback
        probs = np.ones_like(probs, dtype=np.float64) / probs.size
    else:
        probs = probs / s

    raw = probs * float(total)
    base = np.floor(raw).astype(np.int64)
    frac = raw - base

    current = int(base.sum())
    rem = int(total - current)

    # deterministic tie-break: earlier index first
    # For adding: sort by (-frac, +idx)
    # For subtracting: sort by (+frac, +idx) but only where base>0
    if rem > 0:
        order = np.lexsort((np.arange(frac.size), -frac))
        base[order[:rem]] += 1
    elif rem < 0:
        need = -rem
        mask = base > 0
        if mask.any():
            frac2 = frac.copy()
            frac2[~mask] = np.inf  # never subtract from zeros
            order = np.lexsort((np.arange(frac2.size), frac2))
            # subtract from the smallest frac first
            taken = 0
            for idx in order:
                if base[idx] > 0:
                    base[idx] -= 1
                    taken += 1
                    if taken >= need:
                        break
        # if still not enough (extremely unlikely), clamp and fix last element
        # (keeps method total correct)
        base = np.maximum(base, 0)

    # final safety: fix sum exactly
    diff = int(total - base.sum())
    if diff > 0:
        for i in range(diff):
            base[i % base.size] += 1
    elif diff < 0:
        need = -diff
        i = 0
        while need > 0:
            j = i % base.size
            if base[j] > 0:
                base[j] -= 1
                need -= 1
            i += 1

    return base.astype(np.int64)


def _common_prefix_len_msb(a: int, b: int, precision: int) -> int:
    """
    Count common prefix bits (MSB side) between:
      a encoded on `precision` bits
      b encoded on `precision` bits
    """
    sa = format(int(a), f"0{precision}b")
    sb = format(int(b), f"0{precision}b")
    n = 0
    for ca, cb in zip(sa, sb):
        if ca != cb:
            break
        n += 1
    return n


# =========================
# Core AC state (per sentence)
# =========================
class _ACState:
    __slots__ = ("precision", "max_val", "low", "high")

    def __init__(self, precision: int):
        self.precision = int(precision)
        self.max_val = 1 << self.precision
        self.low = 0
        self.high = self.max_val

    def reset(self):
        self.low = 0
        self.high = self.max_val

    @property
    def int_range(self) -> int:
        return int(self.high - self.low)


# single state for this method instance
# (Framework calls reset_sentence() every new sentence)
_STATE = _ACState(AC_PRECISION)


def step(sorted_probs: torch.Tensor,
         sorted_indices: torch.Tensor,
         bit_reader,
         device=None):
    """
    One-token arithmetic decoding step:
      - truncate to top-k candidates
      - map candidate probs to integer sub-intervals within [low, high)
      - read (peek) PRECISION bits as a point in [0, 2^PRECISION)
      - choose token whose cumulative boundary crosses the point
      - compute common MSB prefix of [new_low, new_high-1], consume that many bits
      - renormalize interval by dropping consumed prefix
    """
    _print_once()

    precision = int(AC_PRECISION)
    k = min(int(AC_K), int(sorted_probs.numel()))
    if k <= 0:
        return int(sorted_indices[0].item()), ""

    probs = sorted_probs[:k].detach().to("cpu", dtype=torch.float64).numpy()
    indices = sorted_indices[:k].detach().to("cpu").numpy()

    # reset if interval collapses
    if _STATE.int_range <= 1:
        _STATE.reset()

    cur_low = int(_STATE.low)
    cur_high = int(_STATE.high)
    cur_int_range = int(cur_high - cur_low)
    if cur_int_range <= 1:
        _STATE.reset()
        cur_low = int(_STATE.low)
        cur_high = int(_STATE.high)
        cur_int_range = int(cur_high - cur_low)

    # quantize probabilities into counts summing to cur_int_range
    counts = _quantize_probs_to_counts(probs, cur_int_range)

    # build cumulative boundaries within [cur_low, cur_high)
    cum = np.cumsum(counts, dtype=np.int64)
    if cum.size == 0:
        token_id = int(indices[0])
        return token_id, ""

    # ensure last boundary hits exactly the top
    cum[-1] = cur_int_range
    cum = cum + cur_low  # now boundaries are absolute

    # ---- peek precision bits (do NOT consume yet) ----
    snap = bit_reader.snapshot()
    peek_bits = bit_reader.read_bits(precision)
    bit_reader.restore(snap)

    if peek_bits is None:
        peek_bits = ""
    if len(peek_bits) < precision:
        peek_bits = peek_bits + ("0" * (precision - len(peek_bits)))

    # Interpret as MSB-first integer (equivalent to author's reverse+LSB routine)
    message_idx = int(peek_bits, 2)

    # select first cum boundary > message_idx
    # (if message_idx lands beyond, pick last)
    pos = int(np.searchsorted(cum, message_idx + 1, side="left"))
    if pos >= cum.size:
        pos = int(cum.size - 1)

    new_low = int(cum[pos - 1]) if pos > 0 else cur_low
    new_high = int(cum[pos])

    # if degenerate (can happen with many zeros), fall back
    if new_high <= new_low:
        token_id = int(indices[pos])
        return token_id, ""

    # ---- compute how many bits are now fixed (common MSB prefix) ----
    # use new_high-1 for inclusive top (author code)
    num_bits_encoded = _common_prefix_len_msb(new_low, new_high - 1, precision)

    consumed_bits = bit_reader.read_bits(num_bits_encoded) if num_bits_encoded > 0 else ""
    if consumed_bits is None:
        consumed_bits = ""

    # ---- renormalize: drop prefix, pad with 0/1 ----
    low_bin = format(int(new_low), f"0{precision}b")
    high_bin = format(int(new_high - 1), f"0{precision}b")

    # remove prefix, append num_bits_encoded bits at the end
    low_bin2 = low_bin[num_bits_encoded:] + ("0" * num_bits_encoded)
    high_bin2 = high_bin[num_bits_encoded:] + ("1" * num_bits_encoded)

    _STATE.low = int(low_bin2, 2)
    _STATE.high = int(high_bin2, 2) + 1

    token_id = int(indices[pos])
    return token_id, consumed_bits


class StegoMethod:
    """
    Framework adapter:
      - reset_sentence(): reset interval each new sentence
      - step(): one token decision + consumed bits
    """
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        _print_once()
        # ensure state matches env precision
        global _STATE
        _STATE = _ACState(int(AC_PRECISION))

    def reset_sentence(self):
        _STATE.reset()

    def step(self, sorted_probs, sorted_indices, bit_reader):
        return step(sorted_probs, sorted_indices, bit_reader, self.config.DEVICE)
