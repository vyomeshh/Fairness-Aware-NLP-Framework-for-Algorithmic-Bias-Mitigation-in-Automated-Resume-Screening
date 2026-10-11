"""M5 - cosine similarity (Eq. 3.3) and ranking with seeded tie-breaking."""
from typing import Dict, List, Sequence
import numpy as np


def cosine_scores(cand_vecs: np.ndarray, job_vec: np.ndarray) -> np.ndarray:
    """S = (V_c . V_j) / (||V_c||_2 ||V_j||_2). Zero vectors score 0."""
    cand_vecs = np.atleast_2d(cand_vecs).astype(float)
    job_vec = np.asarray(job_vec, dtype=float).ravel()
    denom = np.linalg.norm(cand_vecs, axis=1) * np.linalg.norm(job_vec)
    dots = cand_vecs @ job_vec
    out = np.zeros_like(dots)
    nz = denom > 0
    out[nz] = dots[nz] / denom[nz]
    return out


def rank_candidates(ids: Sequence[str], scores: Sequence[float], seed: int = 42,
                    decimals: int = 10) -> List[str]:
    """Descending by score. Exact ties (common after masking, when two resumes
    become identical) are broken by a SEEDED random permutation - NOT by id
    order, which would otherwise create an artificial group disparity."""
    ids = list(ids)
    rng = np.random.default_rng(seed)
    tie = rng.permutation(len(ids))
    s = np.round(np.asarray(scores, dtype=float), decimals)
    order = sorted(range(len(ids)), key=lambda i: (-s[i], tie[i]))
    return [ids[i] for i in order]
