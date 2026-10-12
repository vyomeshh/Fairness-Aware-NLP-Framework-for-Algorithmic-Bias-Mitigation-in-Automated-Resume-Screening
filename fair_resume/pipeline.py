"""End-to-end paired pipeline: Condition A (unmasked) vs Condition B (masked).

Only the masking step differs between conditions (IPR-I Sec. 3.6): same job
descriptions, same encoder, same similarity function, same ranking rule, same
seed, same evaluation cases."""
from typing import Dict, List, Set

from .config import MaskConfig
from .m2_parse import Parser
from .m3_mask import mask_text
from .m5_rank import cosine_scores, rank_candidates
from . import m6_metrics as M


def build_conditions(resumes: Dict[str, str], parser: Parser, cfg: MaskConfig):
    """Return (texts_A, texts_B, audit) where audit[rid] is the M3 audit log."""
    texts_a, texts_b, audit = {}, {}, {}
    for rid, t in resumes.items():
        masked, log = mask_text(parser.parse(t), cfg)
        texts_a[rid], texts_b[rid], audit[rid] = t, masked, log
    return texts_a, texts_b, audit


def rank_condition(texts: Dict[str, str], jds: Dict[str, str], encoder, cfg: MaskConfig):
    ids = list(texts)
    cand = encoder.encode([texts[i] for i in ids])
    jd_vecs = encoder.encode([jds[j] for j in jds])
    scores, rankings = {}, {}
    for jv, jid in zip(jd_vecs, jds):
        s = cosine_scores(cand, jv)
        scores[jid] = dict(zip(ids, map(float, s)))
        rankings[jid] = rank_candidates(ids, s, seed=cfg.seed)
    return scores, rankings


def evaluate(rankings: Dict[str, List[str]], groups: Dict[str, int],
             relevant: Dict[str, Set[str]], k: int) -> dict:
    """Y_hat=1 iff the candidate is in the top-k for that job; every
    (candidate, job) pair is one observation of the group selection rate."""
    sel, grp = [], []
    for jid, ranked in rankings.items():
        top = set(ranked[:k])
        for rid in ranked:
            sel.append(int(rid in top))
            grp.append(groups[rid])
    rates = M.selection_rates(sel, grp)
    return {
        "k": k,
        "rate_Z1_reference": rates[1], "rate_Z0_comparison": rates[0],
        "DI": M.disparate_impact(sel, grp),
        "DPD": M.demographic_parity_difference(sel, grp),
        "MRR": M.mean_reciprocal_rank(rankings, relevant),
        "MAP": M.mean_average_precision(rankings, relevant, k=k),
        f"Recall@{k}": M.recall_at_k(rankings, relevant, k),
    }


def run_paired(resumes, jds, groups, relevant, parser: Parser, encoder, cfg: MaskConfig):
    texts_a, texts_b, audit = build_conditions(resumes, parser, cfg)
    # Fit-once-on-union (no-op for pretrained SBERT) -> identical encoder for A and B
    encoder.prepare(list(texts_a.values()) + list(texts_b.values()) + list(jds.values()))
    sa, ra = rank_condition(texts_a, jds, encoder, cfg)
    sb, rb = rank_condition(texts_b, jds, encoder, cfg)
    return {
        "encoder": encoder.name, "parser": parser.backend,
        "texts": {"A": texts_a, "B": texts_b}, "audit": audit,
        "scores": {"A": sa, "B": sb}, "rankings": {"A": ra, "B": rb},
        "metrics": {"A": evaluate(ra, groups, relevant, cfg.top_k),
                    "B": evaluate(rb, groups, relevant, cfg.top_k)},
    }
