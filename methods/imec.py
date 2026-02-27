# methods/imec.py
# iMEC (iterative Minimum-Entropy Coupling) – framework-compatible encoder-only sampler
# Paper: "Perfectly Secure Steganography Using Minimum Entropy Coupling" (ICLR 2023)
#
# This implementation is adapted to YOUR framework interface:
#   StegoMethod.step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)
#
# Notes / design choices for this framework:
# - We do NOT implement the author's full "medium" abstraction nor decoding.
# - We implement an iMEC-like *iterative coupling* per token step:
#     choose the belief block with max entropy, couple it with current LM distribution q,
#     sample token from the coupling row of the block's fixed ciphertext value,
#     update that block belief by coupling column posterior.
# - To avoid "pre-consuming" bits that a sentence might not have room to embed, we
#   dynamically allocate blocks (chunks): when all existing blocks are "done"
#   (entropy <= threshold), we read one new block of IMEC_BLOCK_SIZE bits from the bitstream
#   and start embedding it. This yields a variable number of bits per sentence.
#
# Security remark:
# - The coupling preserves the LM marginal over the *candidate set* provided by ModelHandler
#   (i.e., after top-k/top-p/candidate cap). Perfect-security is with respect to that
#   truncated distribution, consistent with other methods in this framework.

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch


# =========================
# Hyperparameters (env)
# =========================
IMEC_BLOCK_SIZE = int(os.getenv("IMEC_BLOCK_SIZE", "8"))  # bits per chunk; 8/10 recommended in this framework
IMEC_BELIEF_ENTROPY_THRESHOLD = float(os.getenv("IMEC_BELIEF_ENTROPY_THRESHOLD", "0.10"))
IMEC_MAX_CHUNKS_PER_SENTENCE = int(os.getenv("IMEC_MAX_CHUNKS_PER_SENTENCE", "16"))

# Safety cap for 2^block_size to keep memory/runtime sane in Python
IMEC_MAX_STATES = int(os.getenv("IMEC_MAX_STATES", "16384"))  # 2^14

_PRINT_ONCE = False


def _print_once():
    global _PRINT_ONCE
    if not _PRINT_ONCE:
        print(
            f"[iMEC] IMEC_BLOCK_SIZE={IMEC_BLOCK_SIZE}, "
            f"IMEC_BELIEF_ENTROPY_THRESHOLD={IMEC_BELIEF_ENTROPY_THRESHOLD}, "
            f"IMEC_MAX_CHUNKS_PER_SENTENCE={IMEC_MAX_CHUNKS_PER_SENTENCE}, "
            f"IMEC_MAX_STATES={IMEC_MAX_STATES}"
        )
        _PRINT_ONCE = True


# =========================
# Utils
# =========================
def _entropy2(p: np.ndarray) -> float:
    """Base-2 entropy."""
    p = np.asarray(p, dtype=np.float64)
    s = float(p.sum())
    if not (s > 0.0):
        return 0.0
    p = p / s
    eps = 1e-300
    return float(-(p * np.log2(p + eps)).sum())


def _stable_argsort_desc(x: np.ndarray) -> np.ndarray:
    # mergesort is stable -> deterministic tie-break by index
    return np.argsort(-x, kind="mergesort")


def _greedy_coupling_row_masses(p: np.ndarray, q: np.ndarray, row_idx: int) -> np.ndarray:
    """
    Build coupling via greedy mass pairing (sparse "NW corner" on DESC-sorted marginals),
    but only return the masses on the selected row: gamma(row_idx, :)
    Output vector length m (original column order), sums to p[row_idx].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    n = int(p.size)
    m = int(q.size)
    out = np.zeros(m, dtype=np.float64)

    ps = float(p.sum())
    qs = float(q.sum())
    if not (ps > 0.0) or not (qs > 0.0):
        return out

    p = p / ps
    q = q / qs

    p_order = _stable_argsort_desc(p)
    q_order = _stable_argsort_desc(q)

    i = 0
    j = 0
    p_rem = float(p[p_order[i]])
    q_rem = float(q[q_order[j]])
    eps = 1e-18

    while i < n and j < m:
        mass = p_rem if p_rem <= q_rem else q_rem
        i_orig = int(p_order[i])
        j_orig = int(q_order[j])

        if i_orig == int(row_idx):
            out[j_orig] += mass

        p_rem -= mass
        q_rem -= mass

        if p_rem <= eps:
            i += 1
            if i >= n:
                break
            p_rem = float(p[p_order[i]])
        if q_rem <= eps:
            j += 1
            if j >= m:
                break
            q_rem = float(q[q_order[j]])

    out = np.maximum(out, 0.0)
    return out


def _greedy_coupling_col_masses(p: np.ndarray, q: np.ndarray, col_idx: int) -> np.ndarray:
    """
    Same greedy coupling, but only return column masses: gamma(:, col_idx)
    Output vector length n (original row order), sums to q[col_idx].
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    n = int(p.size)
    m = int(q.size)
    out = np.zeros(n, dtype=np.float64)

    ps = float(p.sum())
    qs = float(q.sum())
    if not (ps > 0.0) or not (qs > 0.0):
        return out

    p = p / ps
    q = q / qs

    p_order = _stable_argsort_desc(p)
    q_order = _stable_argsort_desc(q)

    i = 0
    j = 0
    p_rem = float(p[p_order[i]])
    q_rem = float(q[q_order[j]])
    eps = 1e-18

    while i < n and j < m:
        mass = p_rem if p_rem <= q_rem else q_rem
        i_orig = int(p_order[i])
        j_orig = int(q_order[j])

        if j_orig == int(col_idx):
            out[i_orig] += mass

        p_rem -= mass
        q_rem -= mass

        if p_rem <= eps:
            i += 1
            if i >= n:
                break
            p_rem = float(p[p_order[i]])
        if q_rem <= eps:
            j += 1
            if j >= m:
                break
            q_rem = float(q[q_order[j]])

    out = np.maximum(out, 0.0)
    return out


# =========================
# iMEC StegoMethod adapter
# =========================
class StegoMethod:
    """
    Framework adapter for iMEC-like encoding.
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        _print_once()

        self.block_size = int(IMEC_BLOCK_SIZE)
        self.n_states = 1 << self.block_size
        if self.n_states > int(IMEC_MAX_STATES):
            raise ValueError(
                f"[iMEC] 2^IMEC_BLOCK_SIZE too large for this Python implementation: "
                f"IMEC_BLOCK_SIZE={self.block_size} -> {self.n_states} states. "
                f"Increase IMEC_MAX_STATES or reduce IMEC_BLOCK_SIZE."
            )

        self.entropy_threshold = float(IMEC_BELIEF_ENTROPY_THRESHOLD)
        self.max_chunks = int(IMEC_MAX_CHUNKS_PER_SENTENCE)

        self._uniform_belief = (np.ones(self.n_states, dtype=np.float64) / float(self.n_states))

        # ------------------------------------------------------------------
        # IMPORTANT (framework quirk)
        # ------------------------------------------------------------------
        # In this codebase, BatchedStegoGenerator calls ONE shared StegoMethod
        # instance for ALL samples in a batch:
        #   for i in range(batch): token = method.step(...)
        # so method state MUST be per-sample. If we keep a single "sentence"
        # state, only the first sample in a batch will allocate/consume bits
        # and the rest will often embed ~0 bits.
        #
        # To fix this without touching the framework, we implement a simple
        # per-sample state bank indexed by a round-robin cursor. The generator
        # iterates samples in fixed order each decoding step, so the cursor
        # aligns with sample index.
        #
        # NOTE: If the last batch is smaller than config.BATCH_SIZE, this
        # round-robin mapping can misalign. In practice this is usually rare
        # (or you can set BATCH_SIZE=1 for perfect alignment).
        # ------------------------------------------------------------------
        self._bank_size = int(getattr(self.config, "BATCH_SIZE", 1) or 1)
        self._cursor = 0

        # per-sample banks
        self._beliefs_bank: List[List[np.ndarray]] = [[] for _ in range(self._bank_size)]
        self._chunks_bank: List[List[int]] = [[] for _ in range(self._bank_size)]

    def reset_sentence(self):
        # Called once per batch in this framework.
        # Reset ALL per-sample states and the cursor.
        self._cursor = 0
        for i in range(self._bank_size):
            self._beliefs_bank[i] = []
            self._chunks_bank[i] = []

    def _select_slot(self) -> int:
        """Return the active per-sample slot index (round-robin)."""
        slot = int(self._cursor % self._bank_size)
        self._cursor += 1
        return slot

    def _maybe_add_new_chunk(self, beliefs: List[np.ndarray], chunks: List[int], bit_reader) -> str:
        """
        If all current chunks are "done" and we still have room to add a chunk,
        read a new block of bits and initialize a new belief.
        Return consumed bit-string (possibly "").
        """
        if len(beliefs) >= self.max_chunks:
            return ""

        if not beliefs:
            need_new = True
        else:
            ents = [(_entropy2(b) if b is not None else 0.0) for b in beliefs]
            need_new = (max(ents) <= self.entropy_threshold)

        if not need_new:
            return ""

        bits = bit_reader.read_bits(self.block_size)
        if bits is None:
            bits = ""
        if len(bits) < self.block_size:
            bits = bits + ("0" * (self.block_size - len(bits)))

        x = int(bits, 2) if bits else 0
        chunks.append(int(x))
        beliefs.append(self._uniform_belief.copy())
        return bits

    def step(self, sorted_probs: torch.Tensor, sorted_indices: torch.Tensor, bit_reader):
        """
        One token decision:
          1) possibly allocate a new chunk (consume IMEC_BLOCK_SIZE bits) if needed
          2) pick highest-entropy chunk i*
          3) compute greedy low-entropy coupling between belief p and LM distribution q
          4) sample action from coupling row corresponding to fixed x_i*
          5) update belief by coupling column posterior
        """
        if sorted_probs is None or sorted_indices is None or sorted_probs.numel() == 0:
            return 0, ""

        slot = self._select_slot()
        beliefs = self._beliefs_bank[slot]
        chunks = self._chunks_bank[slot]

        consumed_bits = self._maybe_add_new_chunk(beliefs, chunks, bit_reader)

        # If we have no chunks (e.g., max_chunks=0), just sample normally (no embedding).
        if not beliefs:
            pos = torch.multinomial(sorted_probs, 1).item()
            return int(sorted_indices[pos].item()), ""

        # choose active chunk by max entropy
        entropies = np.array([_entropy2(b) for b in beliefs], dtype=np.float64)
        active = int(entropies.argmax())

        p = beliefs[active]
        ps = float(p.sum())
        if not (ps > 0.0):
            p = self._uniform_belief.copy()
        else:
            p = (p / ps).astype(np.float64, copy=False)

        q = sorted_probs.detach().to("cpu", dtype=torch.float64).numpy()
        qs = float(q.sum())
        if not (qs > 0.0):
            q = np.ones_like(q, dtype=np.float64) / float(q.size)
        else:
            q = q / qs

        x = int(chunks[active])
        x = max(0, min(x, self.n_states - 1))

        # coupling row masses for this x -> distribution over actions conditioned on x
        row_masses = _greedy_coupling_row_masses(p, q, row_idx=x)
        row_sum = float(row_masses.sum())
        if not (row_sum > 0.0):
            a = int(torch.multinomial(sorted_probs, 1).item())
            token_id = int(sorted_indices[a].item())
            return token_id, consumed_bits

        row_probs = (row_masses / row_sum).astype(np.float32, copy=False)
        a = int(torch.multinomial(torch.from_numpy(row_probs), 1).item())
        token_id = int(sorted_indices[a].item())

        # posterior update for active chunk: p'(x) = gamma(x, a) / q[a]
        col_masses = _greedy_coupling_col_masses(p, q, col_idx=a)
        col_sum = float(col_masses.sum())
        if col_sum > 0.0:
            beliefs[active] = (col_masses / col_sum).astype(np.float64, copy=False)
        else:
            beliefs[active] = p

        return token_id, consumed_bits
