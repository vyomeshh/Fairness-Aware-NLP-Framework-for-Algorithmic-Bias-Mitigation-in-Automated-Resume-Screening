"""Fairness-aware NLP framework for resume screening - MTE (Interim Report-I) code.

Modules follow Chapter 3 of the IPR-I:
  M1 m1_extract   text extraction + normalization
  M2 m2_parse     NER + dependency parsing -> entity spans + dependency graph
  M3 m3_mask      demographic masking M(.) + audit log
  M4 m4_encode    Sentence-BERT encoder f_theta (offline fallback available)
  M5 m5_rank      cosine similarity + ranking
  M6 m6_metrics   DI, DPD, MRR, Recall@k (+ paired bootstrap CI helper)
"""
__version__ = "0.1.0-mte"
