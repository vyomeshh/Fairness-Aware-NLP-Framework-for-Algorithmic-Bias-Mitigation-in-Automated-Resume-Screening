"""MTE (7th-sem, Interim Report-I) live demo.

Shows the early-development items promised in the IPR-I / paper:
  1. BFS/DFS context inspection on the synthetic dependency graph (Fig. 3.2 / 5)
  2. NER-based masking with audit log, on the IPR-I example sentence
  3. Masking-invariance check on controlled demographic variants
  4. Paired pipeline mechanics (Condition A vs B) on SYNTHETIC data

Run:  python demo_mte.py            (add --parser spacy --encoder sbert once installed)
"""
import argparse, json, sys
from pathlib import Path

from fair_resume.config import MaskConfig
from fair_resume.graph_utils import synthetic_graph
from fair_resume.m2_parse import Parser
from fair_resume.m3_mask import mask_text
from fair_resume.m4_encode import get_encoder, TfidfEncoder
from fair_resume.pipeline import run_paired
from fair_resume.synthetic import build_dataset, VARIANTS, PROFILES

ap = argparse.ArgumentParser()
ap.add_argument("--parser", default="auto", choices=["auto", "spacy", "fallback"])
ap.add_argument("--encoder", default="auto", choices=["auto", "sbert", "tfidf"])
ap.add_argument("--k", type=int, default=8)
args = ap.parse_args()
cfg = MaskConfig(top_k=args.k)
parser = Parser(args.parser)
encoder = TfidfEncoder() if args.encoder == "tfidf" else get_encoder(args.encoder)
hr = lambda t: print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")

hr("0. Backends in use")
print(f"parser  : {parser.backend}   ({'real NER + dependency parse' if parser.backend == 'spacy' else 'OFFLINE FALLBACK - rule-based, no syntax'})")
print(f"encoder : {encoder.name}   ({'pretrained Sentence-BERT' if encoder.name == 'sbert' else 'OFFLINE FALLBACK - lexical TF-IDF stand-in'})")

# ---------------------------------------------------------------- 1
hr("1. Dependency-graph context inspection (Algorithm 1) - synthetic graph, Fig. 3.2")
g = synthetic_graph()
tok = g.tokens
seeds = [0, 5, 7]   # Alex, Delhi, she  (tagged PERSON / LOCATION / PRONOUN)
print("sentence :", " ".join(tok))
print("seeds    :", [tok[i] for i in seeds])
print("BFS      :", [(tok[n], d) for n, d in g.bfs(seeds)])
print("DFS      :", [tok[n] for n in g.dfs(seeds)])
print("depth-1 context of seeds:", [tok[i] for i in g.bounded_neighborhood(seeds, 1)])
print("degree summary (inspection only, NOT a score):", {tok[i]: d for i, d in g.degree_summary().items()})

# ---------------------------------------------------------------- 2
hr("2. NER masking + audit log on the IPR-I example sentence")
sent = "Alex built Python APIs in Delhi and she led testing."
masked, log = mask_text(parser.parse(sent), cfg)
print("T  :", sent)
print("T' :", masked)
print("audit log:")
print(json.dumps(log, indent=2))

# ---------------------------------------------------------------- 3
hr("3. Masking-invariance check on controlled demographic variants")
resumes, jds, groups, relevant = build_dataset()
by_profile = {}
for rid, text in resumes.items():
    m, _ = mask_text(parser.parse(text), cfg)
    by_profile.setdefault(rid.rsplit("_", 1)[0], set()).add(m)
ok = sum(len(v) == 1 for v in by_profile.values())
print(f"profiles whose {len(VARIANTS)} demographic variants become IDENTICAL after masking: {ok}/{len(by_profile)}")
for pid, v in by_profile.items():
    if len(v) > 1:
        print(f"  residual differences in {pid}:"); [print("    ", repr(x)) for x in v]
print("(This verifies the masker on the synthetic variants; it is a component check,")
print(" not evidence of fairness improvement on real data.)")

# ---------------------------------------------------------------- 4
hr("4. Paired pipeline mechanics (A: unmasked  vs  B: masked)  --  SYNTHETIC SMOKE TEST")
print("*** NOT EXPERIMENTAL RESULTS: synthetic data, by-construction relevance, ***")
print("*** fallback components may be in use. Shown only to prove M1-M6 run end to end. ***\n")
res = run_paired(resumes, jds, groups, relevant, parser, encoder, cfg)
keys = list(res["metrics"]["A"])
print(f"{'metric':<22}{'A (unmasked)':>16}{'B (masked)':>16}")
for k_ in keys:
    a, b = res["metrics"]["A"][k_], res["metrics"]["B"][k_]
    print(f"{k_:<22}{a:>16.4f}{b:>16.4f}")
out = Path("outputs"); out.mkdir(exist_ok=True)
(out / "audit_log.json").write_text(json.dumps(res["audit"], indent=2))
(out / "smoke_metrics.json").write_text(json.dumps(
    {"WARNING": "synthetic smoke test - not experimental results",
     "parser": res["parser"], "encoder": res["encoder"], "metrics": res["metrics"]}, indent=2))
print(f"\nwrote {out}/audit_log.json and {out}/smoke_metrics.json")
