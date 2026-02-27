# methods/sparsamp.py
# SparSamp: Efficient Provably Secure Steganography Based on Sparse Sampling
# Framework-compatible implementation (embedding only).
#
# Interface:
#   class StegoMethod:
#       __init__(config, tokenizer)
#       reset_sentence()
#       step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)

from __future__ import annotations

import os
import math
import random
import torch


# =========================
# Hyperparameters (env)
# =========================
SPARSAMP_LM = int(os.getenv("SPARSAMP_LM", "32"))
_SPARSAMP_PRNG_SEED_ENV = os.getenv("SPARSAMP_PRNG_SEED", "")
SPARSAMP_CARRY_ACROSS_SENTENCE = bool(int(os.getenv("SPARSAMP_CARRY_ACROSS_SENTENCE", "0")))

_PRINT_ONCE = False


def _print_once():
    global _PRINT_ONCE
    if not _PRINT_ONCE:
        print(
            f"[SparSamp] SPARSAMP_LM={SPARSAMP_LM}, "
            f"SPARSAMP_PRNG_SEED={'(unset)' if not _SPARSAMP_PRNG_SEED_ENV else _SPARSAMP_PRNG_SEED_ENV}, "
            f"SPARSAMP_CARRY_ACROSS_SENTENCE={int(SPARSAMP_CARRY_ACROSS_SENTENCE)}"
        )
        _PRINT_ONCE = True


def _ceil_int(x: float) -> int:
    return int(math.ceil(x))


def _safe_unit_float(x: float) -> float:
    x = float(x)
    x = x % 1.0
    if x < 0.0:
        x += 1.0
    return x


def _peek_bits(bit_reader, n: int) -> str:
    snap = bit_reader.snapshot()
    bits = bit_reader.read_bits(n)
    bit_reader.restore(snap)
    if bits is None:
        bits = ""
    if len(bits) < n:
        bits = bits + ("0" * (n - len(bits)))
    return bits


def _sample_with_r(probs: torch.Tensor, r_m: float) -> tuple[int, float, float]:
    """
    Algorithm 1 sampling with random number r_m:
      find smallest pos s.t. CDF[pos] > r_m (strict)
    Return:
      pos, SE0, SE1
    """
    probs64 = probs.to(dtype=torch.float64)
    cdf = torch.cumsum(probs64, dim=0)

    # IMPORTANT: r tensor must be on same device as cdf for searchsorted
    r_t = cdf.new_tensor(float(r_m))  # same device & dtype
    pos = int(torch.searchsorted(cdf, r_t, right=True).item())
    if pos >= cdf.numel():
        pos = int(cdf.numel() - 1)

    se1 = float(cdf[pos].item())
    se0 = float(cdf[pos - 1].item()) if pos > 0 else 0.0
    return pos, se0, se1


def _sparse_update(se0: float, se1: float, n_m: int, k_m: int, r: float) -> tuple[int, int]:
    """
    Algorithm 2: sparse(SE_i, N_m, k_m, r_i)
    """
    temp0 = _ceil_int((float(se0) - float(r)) * float(n_m))
    temp1 = _ceil_int((float(se1) - float(r)) * float(n_m))

    if float(k_m) + float(r) * float(n_m) >= float(n_m):
        k_m = int(k_m - n_m - temp0)
    else:
        k_m = int(k_m - temp0)

    n_new = int(temp1 - temp0)
    if n_new <= 0:
        n_new = 1
    return n_new, k_m


class StegoMethod:
    """
    SparSamp implementation for your run_gen_stego.py pipeline.

    We PEEK lm bits at block start (do not consume).
    Only when N_m reaches 1 do we actually consume lm bits and return them,
    so stego.bit matches bitstream consumption even if sentence ends early.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        _print_once()

        self.lm = int(getattr(config, "SPARSAMP_LM", SPARSAMP_LM))
        if self.lm <= 0:
            raise ValueError("SPARSAMP_LM must be > 0")

        # independent PRNG seed (avoid interference with run_gen_stego.py's random)
        if hasattr(config, "SPARSAMP_PRNG_SEED"):
            seed = int(getattr(config, "SPARSAMP_PRNG_SEED"))
        elif _SPARSAMP_PRNG_SEED_ENV:
            seed = int(_SPARSAMP_PRNG_SEED_ENV)
        else:
            base_seed = int(getattr(config, "SEED", 1234) or 1234)
            seed = (base_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
        self.rng = random.Random(seed)

        self.carry_across_sentence = bool(
            getattr(config, "SPARSAMP_CARRY_ACROSS_SENTENCE", SPARSAMP_CARRY_ACROSS_SENTENCE)
        )

        self.N_m = 1
        self.k_m = 0
        self.pending_bits = ""
        self.block_active = False

    def reset_sentence(self):
        if self.carry_across_sentence:
            return
        self.N_m = 1
        self.k_m = 0
        self.pending_bits = ""
        self.block_active = False

    def _start_new_block_peek(self, bit_reader):
        bits = _peek_bits(bit_reader, self.lm)  # DO NOT consume yet
        self.pending_bits = bits
        self.k_m = int(bits, 2)  # MSB-first
        self.N_m = 1 << self.lm
        self.block_active = True

    def step(self, sorted_probs: torch.Tensor, sorted_indices: torch.Tensor, bit_reader):
        if sorted_probs.numel() <= 0:
            return int(sorted_indices[0].item()), ""

        probs = sorted_probs
        if (not torch.isfinite(probs).all()) or float(probs.sum().item()) <= 0.0:
            probs = torch.ones_like(probs, dtype=torch.float32) / probs.numel()
        else:
            probs = probs / probs.sum()

        if (not self.block_active) or self.N_m == 1:
            self._start_new_block_peek(bit_reader)

        r = _safe_unit_float(self.rng.random())
        r_m = _safe_unit_float(r + (float(self.k_m) / float(self.N_m)))

        pos, se0, se1 = _sample_with_r(probs, r_m)
        token_id = int(sorted_indices[pos].item())

        self.N_m, self.k_m = _sparse_update(se0, se1, self.N_m, self.k_m, r)

        if self.N_m == 1 and self.block_active:
            consumed = bit_reader.read_bits(self.lm)  # NOW consume for real
            if consumed is None:
                consumed = ""
            if len(consumed) < self.lm:
                consumed = consumed + ("0" * (self.lm - len(consumed)))

            self.pending_bits = ""
            self.block_active = False
            return token_id, consumed

        return token_id, ""
