# methods/discop.py
# Discop (Ding et al., IEEE S&P 2023): Distribution Copies + Huffman recursion
# Framework-compatible version:
#   StegoMethod.step(sorted_probs, sorted_indices, bit_reader) -> (token_id:int, consumed_bits:str)

from __future__ import annotations

import os
import random
from collections import deque
from typing import List, Tuple

import torch

# =========================
# Hyperparameters (env)
# =========================
# Whether to reseed Discop's private PRNG at the start of each sentence.
# (Recommended in this framework because run_gen_stego may discard sentences and only roll back bitstream.)
DISCOP_RESEED_EACH_SENTENCE = bool(int(os.getenv("DISCOP_RESEED_EACH_SENTENCE", "1")))

_PRINT_ONCE = False


def _print_once():
    global _PRINT_ONCE
    if not _PRINT_ONCE:
        print(f"[Discop] DISCOP_RESEED_EACH_SENTENCE={int(DISCOP_RESEED_EACH_SENTENCE)}")
        _PRINT_ONCE = True


def _rotate_right(a: float, d: float, e: float) -> float:
    """Rotate a to the right by d within [0, e)."""
    b = a + d
    if b >= e:
        b -= e
    return b


def _build_huffman_arrays_from_desc_probs(
    probs_desc: List[float],
    token_ids_desc: List[int],
) -> Tuple[List[float], List[int], List[int], int, int]:
    """
    Build Huffman tree using the paper's linear-time two-queue method (Algorithm 4),
    implemented with index arrays (no Node objects).

    Input:
        probs_desc/token_ids_desc are DESC-sorted (as provided by ModelHandler.process_probs()).

    Returns:
        prob:        list[float] length (2n-1), subtree probability for each node index
        left_child:  list[int] length (2n-1), left child index for internal nodes, -1 for leaves
        right_child: list[int] length (2n-1), right child index for internal nodes, -1 for leaves
        root_idx:    int, root node index
        n_leaves:    int, number of leaves (original candidate count)
    """
    n = int(len(probs_desc))
    if n <= 0:
        return [1.0], [-1], [-1], 0, 0
    if n == 1:
        # single leaf only
        return [float(probs_desc[0])], [-1], [-1], 0, 1

    # Convert DESC -> ASC (required for linear Huffman)
    probs_asc = list(reversed(probs_desc))
    token_ids_asc = list(reversed(token_ids_desc))

    total_nodes = 2 * n - 1
    prob = [0.0] * total_nodes
    left_child = [-1] * total_nodes
    right_child = [-1] * total_nodes

    # fill leaves
    for i in range(n):
        p = float(probs_asc[i])
        if not (p >= 0.0) or not (p == p):  # NaN guard
            p = 0.0
        prob[i] = p

    # queues hold node indices; since leaves are ASC, q1.front() is current minimum
    q1 = deque(range(n))
    q2: deque[int] = deque()

    def get_min() -> int:
        if q1 and q2:
            # deterministic tie-break: prefer q1 when equal
            if prob[q1[0]] <= prob[q2[0]]:
                return q1.popleft()
            return q2.popleft()
        if q1:
            return q1.popleft()
        return q2.popleft()

    # create internal nodes in increasing index order
    for new_idx in range(n, total_nodes):
        a = get_min()
        b = get_min()
        left_child[new_idx] = a
        right_child[new_idx] = b
        prob[new_idx] = prob[a] + prob[b]
        q2.append(new_idx)

    root_idx = total_nodes - 1
    # return also token_ids_asc so leaf index -> token_id mapping is known by caller
    # (we return token_ids_asc as "leaf_token_ids")
    return prob, left_child, right_child, token_ids_asc, root_idx, n


def _discop_sample_one_token(
    sorted_probs: torch.Tensor,
    sorted_indices: torch.Tensor,
    bit_reader,
    rng: random.Random,
) -> Tuple[int, str]:
    """
    Discop sampling for ONE next token:
      - build Huffman tree from current distribution
      - traverse root->leaf; at each internal node embed up to 1 bit via distribution copies
      - return (token_id, consumed_bits_str)
    """
    # Convert to python lists (DESC order already).
    probs_desc = sorted_probs.detach().to("cpu", dtype=torch.float32).tolist()
    token_ids_desc = sorted_indices.detach().to("cpu").tolist()

    if not probs_desc or not token_ids_desc:
        # extremely defensive fallback
        return int(sorted_indices[0].item()), ""

    if len(probs_desc) == 1:
        return int(token_ids_desc[0]), ""

    # Build Huffman arrays (linear-time method).
    prob, left_child, right_child, leaf_token_ids_asc, root_idx, n_leaves = _build_huffman_arrays_from_desc_probs(
        probs_desc, token_ids_desc
    )

    consumed_bits: List[str] = []

    node = int(root_idx)
    while node >= n_leaves:
        e = float(prob[node])
        if not (e > 0.0):
            # degenerate; fall back to most likely token
            return int(token_ids_desc[0]), "".join(consumed_bits)

        l = int(left_child[node])
        r = int(right_child[node])
        sep = float(prob[l])  # left interval length within [0, e)

        # r ~ U[0, e)
        r0 = rng.random() * e
        r1 = _rotate_right(r0, 0.5 * e, e)

        next0 = l if r0 < sep else r
        next1 = l if r1 < sep else r

        if next0 != next1:
            b = bit_reader.read_bits(1)
            if b is None or len(b) == 0:
                b = "0"
            else:
                b = b[0]  # ensure exactly 1 char

            consumed_bits.append(b)
            node = next0 if b == "0" else next1
        else:
            # disputed range: embed 0 bit, must go to the unique child
            node = next0

    # leaf reached
    token_id = int(leaf_token_ids_asc[node])
    return token_id, "".join(consumed_bits)


class StegoMethod:
    """
    Framework adapter for Discop:
      - reset_sentence(): optionally reseed an internal PRNG each sentence attempt
      - step(): run one Discop token decision (may consume multiple bits)
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        _print_once()

        self._base_seed = int(getattr(config, "SEED", 0) or 0)
        self._sent_nonce = 0

        # private PRNG (do NOT use global random, to avoid interfering with EMBED_SKIP_PROB logic)
        self._rng = random.Random(self._base_seed)

    def reset_sentence(self):
        if not DISCOP_RESEED_EACH_SENTENCE:
            return
        self._sent_nonce += 1
        # simple deterministic mixing to avoid identical PRNG streams across sentences
        seed64 = (self._base_seed ^ (self._sent_nonce * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
        self._rng.seed(seed64)

    def step(self, sorted_probs, sorted_indices, bit_reader):
        return _discop_sample_one_token(sorted_probs, sorted_indices, bit_reader, self._rng)
