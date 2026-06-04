"""
Lightweight WER and text normalization for TSA evaluations.
Uses jiwer if available, else word-level edit distance.
"""
from __future__ import annotations

import re
import string
from typing import Tuple


# Minimum reference token count for tWER/cpWER (exclude item from aggregation if below)
MIN_REF_TOKENS_TWER_CPWER = 4


def normalize_twer_cpwer(text: str) -> str:
    """
    Normalization for tWER/cpWER pipeline (paper tables).
    Rules: uppercase; remove punctuation; remove tokens containing digits; collapse spaces; tokenize by whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    t = text.upper().strip()
    # Remove punctuation (replace with space so we don't glue words)
    for p in string.punctuation:
        t = t.replace(p, " ")
    # Tokenize, drop tokens that contain any digit, rejoin
    tokens = t.split()
    tokens = [w for w in tokens if w and not any(c in "0123456789" for c in w)]
    return " ".join(tokens)


def normalize_for_wer(text: str) -> str:
    """English-oriented: upper, collapse whitespace, remove (()) and <> tags."""
    if not text or not isinstance(text, str):
        return ""
    t = text.upper().strip()
    t = re.sub(r"\(\)", " ", t)
    t = re.sub(r"\([^)]*\)", " ", t)  # e.g. (()) or (noise)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _edit_distance_counts(ref_words: list[str], hyp_words: list[str]) -> Tuple[int, int, int]:
    """Returns (substitutions, deletions, insertions)."""
    if not ref_words and not hyp_words:
        return 0, 0, 0
    if not ref_words:
        return 0, 0, len(hyp_words)
    if not hyp_words:
        return 0, len(ref_words), 0
    rows, cols = len(ref_words) + 1, len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]
    ops = [[(0, 0, 0)] * cols for _ in range(rows)]
    for i in range(1, rows):
        dp[i][0] = i
        ops[i][0] = (0, i, 0)
    for j in range(1, cols):
        dp[0][j] = j
        ops[0][j] = (0, 0, j)
    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = ops[i - 1][j - 1]
                continue
            sub_c = dp[i - 1][j - 1] + 1
            del_c = dp[i - 1][j] + 1
            ins_c = dp[i][j - 1] + 1
            best = min(sub_c, del_c, ins_c)
            if best == sub_c:
                s, d, ins = ops[i - 1][j - 1]
                ops[i][j] = (s + 1, d, ins)
            elif best == del_c:
                s, d, ins = ops[i - 1][j]
                ops[i][j] = (s, d + 1, ins)
            else:
                s, d, ins = ops[i][j - 1]
                ops[i][j] = (s, d, ins + 1)
            dp[i][j] = best
    return ops[-1][-1]


def compute_wer_single(ref: str, hyp: str, normalize: bool = True) -> Tuple[float | None, int, int, int, int]:
    """
    Returns (wer, n_ref_words, substitutions, deletions, insertions).
    wer is None if n_ref_words == 0.
    """
    if normalize:
        ref = normalize_for_wer(ref)
        hyp = normalize_for_wer(hyp)
    ref_w = ref.split()
    hyp_w = hyp.split()
    n_ref = len(ref_w)
    if n_ref == 0:
        return None, 0, 0, 0, 0
    s, d, ins = _edit_distance_counts(ref_w, hyp_w)
    err = s + d + ins
    return err / n_ref, n_ref, s, d, ins


def compute_wer_percent_twer_cpwer(
    ref: str,
    hyp: str,
    min_ref_tokens: int = MIN_REF_TOKENS_TWER_CPWER,
) -> Tuple[float | None, int]:
    """
    WER for tWER/cpWER pipeline: normalize with normalize_twer_cpwer, then WER = (S+D+I)/N_ref.
    Returns (wer_percent, n_ref_tokens). wer_percent is None if ref is empty or n_ref_tokens < min_ref_tokens.
    """
    ref_n = normalize_twer_cpwer(ref)
    hyp_n = normalize_twer_cpwer(hyp)
    ref_w = ref_n.split()
    n_ref = len(ref_w)
    if n_ref == 0 or n_ref < min_ref_tokens:
        return None, n_ref
    hyp_w = hyp_n.split()
    s, d, ins = _edit_distance_counts(ref_w, hyp_w)
    err = s + d + ins
    wer_frac = err / n_ref
    return 100.0 * wer_frac, n_ref


def compute_tcpwer(
    ref_target: str,
    ref_nontarget: str,
    hyp_target: str,
    hyp_nontarget: str,
    normalize: bool = True,
) -> Tuple[float | None, int, int, int, int]:
    """
    Time-constrained permutation WER over two speakers: combine ref/hyp per speaker, then total errors / total ref words.
    Returns (tcpwer, total_ref_words, total_subs, total_dels, total_ins).
    """
    _, n_t, s_t, d_t, i_t = compute_wer_single(ref_target, hyp_target, normalize)
    _, n_n, s_n, d_n, i_n = compute_wer_single(ref_nontarget, hyp_nontarget, normalize)
    total_ref = n_t + n_n
    if total_ref == 0:
        return None, 0, 0, 0, 0
    total_err = (s_t + d_t + i_t) + (s_n + d_n + i_n)
    return total_err / total_ref, total_ref, s_t + s_n, d_t + d_n, i_t + i_n
