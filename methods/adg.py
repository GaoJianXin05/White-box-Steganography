# methods/adg.py  –  ADG (Adaptive Dynamic Grouping)
# Faithful to: Zhang et al., ACL 2021, Algorithm 1 + Recursion/Pruning
# Notes:
# - ADG_R_MAX is controlled ONLY here (env ADG_R_MAX), not in GlobalConfig.
# - Default ADG_R_MAX = 5.
# - If env ADG_R_MAX is invalid -> fallback to default.
# - If env ADG_R_MAX < 0 -> treated as 0 (disable cap), to avoid surprises/crashes.

import os
import math
import torch
from typing import List, Tuple, Optional

_DEFAULT_ADG_R_MAX = 5
_ADG_R_MAX_PRINTED = False


def _get_adg_r_max() -> int:
    raw = os.getenv("ADG_R_MAX", None)
    if raw is None or raw == "":
        return _DEFAULT_ADG_R_MAX
    try:
        v = int(raw)
    except Exception:
        v = _DEFAULT_ADG_R_MAX

    # robust: negative -> disable cap (0)
    if v < 0:
        v = 0
    return v


def _bits_lsb_to_int(bits_str: str) -> int:
    """LSB-first bit string -> integer. Example: '011' -> 6."""
    return int(bits_str[::-1], 2) if bits_str else 0


def _calc_bit_depth(p_max: float, n_candidates: int) -> int:
    """
    Compute ADG bit depth r.

    Paper: r = floor(-log2(p_max)), with constraints:
      - r >= 1
      - 2^r <= n_candidates
      - optional cap r <= ADG_R_MAX if ADG_R_MAX > 0
    """
    global _ADG_R_MAX_PRINTED
    rmax = _get_adg_r_max()
    if not _ADG_R_MAX_PRINTED:
        print(f"[ADG] ADG_R_MAX (effective) = {rmax}  (default={_DEFAULT_ADG_R_MAX}, env=ADG_R_MAX)")
        _ADG_R_MAX_PRINTED = True

    if n_candidates <= 1 or not (p_max > 0.0):
        return 0

    r = int(math.floor(-math.log2(p_max)))
    if r < 1:
        return 0

    max_r_by_n = int(math.floor(math.log2(n_candidates)))
    r = min(r, max_r_by_n)

    if rmax > 0:
        r = min(r, rmax)

    return r


def _find_nearest_desc(probs: List[float], target: float) -> int:
    """
    Binary search in a DESC-sorted list to find index with value nearest to target.
    Tie-break: prefer higher prob (smaller index).
    """
    n = len(probs)
    if n <= 1:
        return 0
    if target >= probs[0]:
        return 0
    if target <= probs[-1]:
        return n - 1

    lo, hi = 0, n - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        v = probs[mid]
        if v == target:
            return mid
        if v > target:
            lo = mid
        else:
            hi = mid

    if abs(probs[lo] - target) <= abs(probs[hi] - target):
        return lo
    return hi


def _build_one_group(
    rem_probs: List[float],
    rem_indices: List[int],
    mean: float,
) -> Tuple[List[float], List[int], float]:
    """
    Algorithm 1 greedy group construction (one group).
    Mutates rem_probs / rem_indices via pop.
    Returns (group_probs, group_indices, group_sum).
    """
    g_probs = [rem_probs.pop(0)]
    g_idxs = [rem_indices.pop(0)]
    g_sum = g_probs[0]

    while rem_probs and g_sum < mean:
        delta = mean - g_sum
        j = _find_nearest_desc(rem_probs, delta)
        cand = rem_probs[j]

        # Paper condition: cand - delta < delta  <=>  cand < 2*delta
        if (cand - delta) < delta:
            g_sum += cand
            g_probs.append(rem_probs.pop(j))
            g_idxs.append(rem_indices.pop(j))
        else:
            break

    return g_probs, g_idxs, g_sum


def step(sorted_probs, sorted_indices, bit_reader, device=None):
    """
    ADG one-token stegosampling.

    Args:
        sorted_probs: 1-D tensor (desc), probabilities
        sorted_indices: 1-D tensor token ids aligned with sorted_probs
        bit_reader: BitStreamReader

    Returns:
        token_id: int
        consumed_bits: str
    """
    probs: List[float] = sorted_probs.detach().to("cpu", dtype=torch.float64).tolist()
    indices: List[int] = sorted_indices.detach().to("cpu").tolist()

    consumed_parts: List[str] = []

    while len(probs) > 1 and probs[0] <= 0.5:
        r = _calc_bit_depth(probs[0], len(probs))
        if r <= 0:
            break
        u = 1 << r

        bits_str = bit_reader.read_bits(r)
        consumed_parts.append(bits_str)
        target_gid = _bits_lsb_to_int(bits_str)

        rem_probs = probs
        rem_indices = indices
        rem_sum = float(sum(rem_probs))
        mean = rem_sum / u

        if target_gid == u - 1:
            # last group = remainder
            for gi in range(u - 1):
                _, _, g_s = _build_one_group(rem_probs, rem_indices, mean)
                rem_sum -= g_s
                remain_groups = u - gi - 1
                mean = (rem_sum / remain_groups) if remain_groups > 0 else 0.0

            sel_probs = rem_probs
            sel_indices = rem_indices
            already_sorted_desc = True
        else:
            sel_probs: Optional[List[float]] = None
            sel_indices: Optional[List[int]] = None
            already_sorted_desc = False

            for gi in range(target_gid + 1):
                g_p, g_i, g_s = _build_one_group(rem_probs, rem_indices, mean)
                if gi == target_gid:
                    sel_probs, sel_indices = g_p, g_i
                    break
                rem_sum -= g_s
                remain_groups = u - gi - 1
                mean = (rem_sum / remain_groups) if remain_groups > 0 else 0.0

            if not sel_probs or not sel_indices:
                break

        s = float(sum(sel_probs))
        if not (s > 0.0):
            probs = [1.0 / len(sel_probs)] * len(sel_probs)
        else:
            inv = 1.0 / s
            probs = [p * inv for p in sel_probs]
        indices = sel_indices

        if not already_sorted_desc:
            paired = sorted(zip(probs, indices), key=lambda x: (-x[0], x[1]))
            probs = [p for p, _ in paired]
            indices = [i for _, i in paired]

    if not probs:
        return int(sorted_indices[0].item()), "".join(consumed_parts)

    p_tensor = torch.tensor(probs, dtype=torch.float32)
    ps = p_tensor.sum()
    if (not torch.isfinite(ps)) or ps <= 0:
        p_tensor = torch.ones_like(p_tensor) / p_tensor.numel()
    else:
        p_tensor = p_tensor / ps

    pick = int(torch.multinomial(p_tensor, 1).item())
    token_id = int(indices[pick])
    return token_id, "".join(consumed_parts)


adg_step = step


class StegoMethod:
    """
    Unified method wrapper:
      - __init__(config, tokenizer): keep compatibility with method loader
      - reset_sentence(): ADG is stateless across sentences
      - step(...): returns (token_id, consumed_bits)
    """

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

    def reset_sentence(self):
        pass

    def step(self, sorted_probs, sorted_indices, bit_reader):
        token_id, bits = step(sorted_probs, sorted_indices, bit_reader, device=getattr(self.config, "DEVICE", None))
        return token_id, bits
