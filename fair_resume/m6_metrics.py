"""M6 - fairness and utility metrics (Eqs. 3.4-3.7) + paired bootstrap CI."""
import math
from typing import Dict, Iterable, List, Sequence, Set
import numpy as np


def selection_rates(selected: Sequence[int], groups: Sequence[int]) -> Dict[int, float]:
    """P(Y_hat=1 | Z=z) for z in {0,1}. NaN if a group is empty."""
    selected, groups = np.asarray(selected), np.asarray(groups)
    out = {}
    for z in (0, 1):
        m = groups == z
        out[z] = float(selected[m].mean()) if m.any() else float("nan")
    return out


def disparate_impact(selected, groups) -> float:
    """DI = P(Y=1|Z=0) / P(Y=1|Z=1). NaN (undefined) if the reference-group
    selection rate is zero - callers must document how this case is handled."""
    r = selection_rates(selected, groups)
    if r[1] == 0 or math.isnan(r[1]):
        return float("nan")
    return r[0] / r[1]


def demographic_parity_difference(selected, groups) -> float:
    """DPD = P(Y=1|Z=0) - P(Y=1|Z=1) (signed)."""
    r = selection_rates(selected, groups)
    return r[0] - r[1]


def mean_reciprocal_rank(rankings: Dict[str, List[str]], relevant: Dict[str, Set[str]]) -> float:
    """MRR = (1/|Q|) sum_q 1/rank_q ; a query with no relevant item retrieved
    contributes 0 (rank = infinity). Queries with no relevance labels are skipped."""
    rr = []
    for q, ranked in rankings.items():
        rel = relevant.get(q, set())
        if not rel:
            continue
        pos = next((i + 1 for i, c in enumerate(ranked) if c in rel), None)
        rr.append(1.0 / pos if pos else 0.0)
    return float(np.mean(rr)) if rr else float("nan")


def recall_at_k(rankings: Dict[str, List[str]], relevant: Dict[str, Set[str]], k: int) -> float:
    """Recall@k = |Relevant ∩ Top_k| / |Relevant|, averaged over queries that
    have at least one relevant item."""
    vals = []
    for q, ranked in rankings.items():
        rel = relevant.get(q, set())
        if not rel:
            continue
        vals.append(len(rel & set(ranked[:k])) / len(rel))
    return float(np.mean(vals)) if vals else float("nan")


def average_precision(ranked: Sequence[str], relevant: Set[str], k: int = None) -> float:
    """Average Precision (AP) for a query up to rank k."""
    if not relevant:
        return float("nan")
    items = ranked[:k] if k is not None else ranked
    hits, score = 0, 0.0
    for i, item in enumerate(items):
        if item in relevant:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(relevant), len(items))
    return (score / denom) if denom > 0 else 0.0


def mean_average_precision(rankings: Dict[str, List[str]], relevant: Dict[str, Set[str]],
                           k: int = None) -> float:
    """Mean Average Precision (MAP) across queries with relevant items."""
    scores = []
    for q, ranked in rankings.items():
        rel = relevant.get(q, set())
        if rel:
            scores.append(average_precision(ranked, rel, k=k))
    return float(np.mean(scores)) if scores else float("nan")


def paired_bootstrap_ci(diffs: Iterable[float], n_boot: int = 2000, alpha: float = 0.05,
                        seed: int = 42):
    """Mean of paired differences (masked - baseline) with a percentile bootstrap
    CI. A helper for the later experiment stage; it is not evidence by itself."""
    d = np.asarray(list(diffs), dtype=float)
    if d.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), (float(lo), float(hi))
