"""M3 - demographic masking T' = M(T) and audit log (Eq. 3.1)."""
from typing import List, Tuple

from .config import MaskConfig
from .m2_parse import ParsedDoc, Entity


def _resolve_overlaps(ents: List[Entity]) -> List[Entity]:
    """Keep earliest/longest span when two spans overlap."""
    out: List[Entity] = []
    for e in sorted(ents, key=lambda x: (x.start, -(x.end - x.start))):
        if out and e.start < out[-1].end:
            continue
        out.append(e)
    return out


def mask_text(doc: ParsedDoc, cfg: MaskConfig = MaskConfig()) -> Tuple[str, List[dict]]:
    """Replace every span in cfg.categories by its neutral token.

    Returns (T', audit_log). Offsets in the audit log refer to the ORIGINAL T so
    every replacement can be reviewed. Text outside selected spans is untouched
    (skills, tools, qualifications are preserved by construction)."""
    selected = _resolve_overlaps([e for e in doc.entities if e.label in cfg.categories])
    pieces, log, cursor = [], [], 0
    for e in selected:
        repl = cfg.neutral_tokens[e.label]
        pieces.append(doc.text[cursor:e.start])
        pieces.append(repl)
        cursor = e.end
        ctx_ids = doc.graph.bounded_neighborhood(e.token_ids, cfg.context_depth)
        head_dep = [(doc.tokens[i].dep, doc.tokens[doc.tokens[i].head].text)
                    for i in e.token_ids if doc.has_syntax and doc.tokens[i].head >= 0]
        log.append({
            "label": e.label, "text": e.text, "start": e.start, "end": e.end,
            "replacement": repl, "source": e.source,
            "dependency": head_dep[0][0] if head_dep else None,
            "head": head_dep[0][1] if head_dep else None,
            "context_tokens": [doc.tokens[i].text for i in ctx_ids],
            "context_kind": "dependency" if doc.has_syntax else "linear-adjacency (no parser)",
        })
    pieces.append(doc.text[cursor:])
    return "".join(pieces), log
