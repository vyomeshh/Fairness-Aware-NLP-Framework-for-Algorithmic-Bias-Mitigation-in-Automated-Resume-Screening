"""Rank a folder of resumes (pdf/docx/txt) against one job description under
Condition A (unmasked) and B (masked); write rankings + audit log.

  python -m fair_resume.cli --resumes ./resumes --jd jd.txt --out ./out
Fairness/utility metrics need group labels and relevance judgments, which are
not available for arbitrary files, so they are not computed here."""
import argparse, csv, json
from pathlib import Path

from .config import MaskConfig
from .m1_extract import extract_text, normalize_text
from .m2_parse import Parser
from .m4_encode import get_encoder
from .pipeline import build_conditions, rank_condition


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--resumes", required=True)
    ap.add_argument("--jd", required=True, help="job description (.txt/.pdf/.docx)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--parser", default="auto", choices=["auto", "spacy", "fallback"])
    ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "tfidf"])
    ap.add_argument("--k", type=int, default=5)
    a = ap.parse_args()

    cfg = MaskConfig(top_k=a.k)
    files = sorted(p for p in Path(a.resumes).iterdir() if p.suffix.lower() in {".pdf", ".docx", ".txt"})
    resumes = {p.stem: extract_text(p) for p in files}
    jds = {"JD": extract_text(a.jd)}
    parser = Parser(a.parser)
    enc = get_encoder("auto" if a.encoder == "auto" else ("sbert" if a.encoder == "sbert" else "tfidf"))
    ta, tb, audit = build_conditions(resumes, parser, cfg)
    enc.prepare(list(ta.values()) + list(tb.values()) + list(jds.values()))
    sa, ra = rank_condition(ta, jds, enc, cfg)
    sb, rb = rank_condition(tb, jds, enc, cfg)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "rankings.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["candidate", "rank_A", "score_A", "rank_B", "score_B"])
        for rid in resumes:
            w.writerow([rid, ra["JD"].index(rid) + 1, round(sa["JD"][rid], 4),
                        rb["JD"].index(rid) + 1, round(sb["JD"][rid], 4)])
    (out / "audit_log.json").write_text(json.dumps(audit, indent=2))
    print(f"parser={parser.backend} encoder={enc.name}; wrote {out}/rankings.csv and audit_log.json")


if __name__ == "__main__":
    main()
