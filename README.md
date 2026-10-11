# Fairness-Aware NLP Framework for Resume Screening - MTE code (Interim Report-I)

Galgotias University, SCSE - Capstone, 7th Sem MTE. Code for the pipeline defined in
IPR-I Chapter 3 and the research paper (Sec. V-VI). Module names M1-M6 match the report.

## What the manual asks for at 7th-sem MTE
Literature review + problem definition + **early development** (Chapters 1-2). This repo is the
early development: every module is implemented and unit-tested, but **no experimental results
are produced or claimed** (consistent with the IPR-I and paper wording).

| Report item | File |
|---|---|
| M1 text extraction / normalization (T) | `fair_resume/m1_extract.py` |
| M2 NER + dependency parsing, graph G=(V,E) | `fair_resume/m2_parse.py`, `graph_utils.py` |
| Algorithm 1 (BFS/DFS context inspection, bounded neighbourhood) | `graph_utils.py` |
| M3 masking T' = M(T) (Eq. 3.1) + audit log | `fair_resume/m3_mask.py` |
| M4 Sentence-BERT f_theta (Eq. 3.2) | `fair_resume/m4_encode.py` |
| M5 cosine similarity (3.3) + ranking | `fair_resume/m5_rank.py` |
| M6 DI, DPD, MRR, Recall@k (3.4-3.7) + paired bootstrap CI | `fair_resume/m6_metrics.py` |
| Condition A vs B paired run (Sec. 3.6) | `fair_resume/pipeline.py` |
| Controlled demographic variants (Sec. 3.6) | `fair_resume/synthetic.py` |
| Live demo for the review | `demo_mte.py` |
| Tests (25) | `tests/test_mte.py` |

## Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m unittest discover -s tests -v      # 25 tests
python demo_mte.py                           # auto-uses spaCy + SBERT if installed
python demo_mte.py --parser spacy --encoder sbert   # force real backends (fails loudly if missing)
python -m fair_resume.cli --resumes ./resumes --jd jd.txt --out ./out   # your own files
```

## Backends - IMPORTANT
The code was written/tested in an offline sandbox, so it has two modes:

* **Real** (spaCy `en_core_web_sm` + `sentence-transformers/all-MiniLM-L6-v2`) - the one the report describes.
  *Not executed in the sandbox* - run the demo once on your machine with `--parser spacy --encoder sbert`
  before the review, and note the invariance count in demo section 3 (statistical NER may miss entities;
  that is an honest finding, not a bug).
* **Fallback** (gazetteer matcher + TF-IDF) - lets everything run offline. The demo prints which backend
  is active. Never report research-question results with the fallback.

## Design decisions worth mentioning to the examiner
* **Seeded tie-breaking** (`m5_rank.py`): after masking, resumes that differ only in name/city/pronoun become
  identical and tie exactly. Breaking ties by id order would manufacture a group disparity, so ties use a
  seeded permutation (seed is a recorded configurable parameter, Table 3.3).
* **Neutral "they/them" is not masked**; only gendered pronouns are.
* **DI undefined** when the reference-group rate is 0 -> returns NaN (must be documented per IPR Sec. 3.5).
* **Audit log** stores original-text offsets, replacement, dependency role/head and BFS depth-1 context
  for every masked span. With the fallback parser the context is linear adjacency, and says so.
* Graph degree/BFS output is for context review only - never a demographic or employability score.

## Honest status (say this at the MTE)
Done: modules M1-M6, paired pipeline, tests, synthetic variant generator.
Not done (ETE / Sem-8): real dataset + licence, relevance labels, group definition Z, real-NER evaluation,
statistical tests on real data, React/FastAPI/PostgreSQL/vector-store prototype.
The synthetic smoke test (demo section 4) is deliberately easy: DI is already 1.0 for the unmasked baseline
because TF-IDF barely reacts to names. It proves the plumbing, **not** that masking helps.
