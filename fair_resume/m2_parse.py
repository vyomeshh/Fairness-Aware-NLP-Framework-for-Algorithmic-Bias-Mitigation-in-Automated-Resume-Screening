"""M2 - NER + dependency parsing -> entity spans and dependency graph.

Two backends:
  * "spacy"    : statistical NER + dependency parser (en_core_web_sm). PRIMARY.
  * "fallback" : offline rule/gazetteer matcher with NO syntactic parser. Its
                 "graph" only links adjacent tokens (label "adj"). It exists so
                 the pipeline can run/test without model downloads; it is NOT a
                 substitute for NER in the experiments.
"""
import re
import warnings
from dataclasses import dataclass, field
from typing import List

from . import config
from .graph_utils import DependencyGraph

SPACY_LABEL_MAP = {"PERSON": "PERSON", "GPE": "LOCATION", "LOC": "LOCATION"}


@dataclass
class Entity:
    start: int            # char offsets in the ORIGINAL text T
    end: int
    text: str
    label: str            # PERSON | LOCATION | PRONOUN
    token_ids: List[int]
    source: str           # "spacy" | "fallback"


@dataclass
class Token:
    i: int
    text: str
    start: int
    end: int
    pos: str = ""
    dep: str = ""
    head: int = -1


@dataclass
class ParsedDoc:
    text: str
    tokens: List[Token]
    entities: List[Entity]
    graph: DependencyGraph
    backend: str
    has_syntax: bool = field(default=False)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019\-]*|\d+|[^\w\s]")


class Parser:
    def __init__(self, backend: str = "auto", model: str = "en_core_web_sm"):
        self.nlp = None
        self.backend = "fallback"
        if backend in ("auto", "spacy"):
            try:
                import spacy
                self.nlp = spacy.load(model)
                self.backend = "spacy"
            except Exception as exc:               # missing package or model
                if backend == "spacy":
                    raise RuntimeError(f"spaCy backend unavailable: {exc}")
                warnings.warn("spaCy/en_core_web_sm not available - using the "
                              "rule-based FALLBACK parser (no real NER/dependency "
                              "parsing). Do not report experimental results with it.")

    # ------------------------------------------------------------------
    def parse(self, text: str) -> ParsedDoc:
        return self._parse_spacy(text) if self.backend == "spacy" else self._parse_fallback(text)

    # ---- spaCy ---------------------------------------------------------
    def _parse_spacy(self, text: str) -> ParsedDoc:
        doc = self.nlp(text)
        tokens = [Token(t.i, t.text, t.idx, t.idx + len(t.text), t.pos_, t.dep_, t.head.i)
                  for t in doc]
        edges = [(t.head.i, t.i, t.dep_) for t in doc if t.head.i != t.i]
        ents: List[Entity] = []
        covered = set()
        for e in doc.ents:
            label = SPACY_LABEL_MAP.get(e.label_)
            if label:
                ids = list(range(e.start, e.end))
                ents.append(Entity(e.start_char, e.end_char, e.text, label, ids, "spacy"))
                covered.update(ids)
        for t in doc:
            if t.pos_ == "PRON" and t.lower_ in config.GENDERED_PRONOUNS and t.i not in covered:
                ents.append(Entity(t.idx, t.idx + len(t.text), t.text, "PRONOUN", [t.i], "spacy"))
        ents.sort(key=lambda e: e.start)
        return ParsedDoc(text, tokens, ents, DependencyGraph([t.text for t in tokens], edges),
                         "spacy", True)

    # ---- fallback ------------------------------------------------------
    def _parse_fallback(self, text: str) -> ParsedDoc:
        toks = [Token(i, m.group(), m.start(), m.end())
                for i, m in enumerate(_TOKEN_RE.finditer(text))]
        n = len(toks)
        ents: List[Entity] = []
        i = 0
        while i < n:
            matched = self._match_at(text, toks, i)
            if matched:
                label, j = matched                    # tokens i..j inclusive
                ents.append(Entity(toks[i].start, toks[j].end, text[toks[i].start:toks[j].end],
                                   label, list(range(i, j + 1)), "fallback"))
                i = j + 1
            else:
                i += 1
        edges = [(k, k + 1, "adj") for k in range(n - 1)]   # linear context, NOT syntax
        return ParsedDoc(text, toks, ents, DependencyGraph([t.text for t in toks], edges),
                         "fallback", False)

    @staticmethod
    def _only_space(text, a, b):
        return text[a:b].strip(" \t") == "" and "\n" not in text[a:b]

    def _match_at(self, text, toks, i):
        n = len(toks)
        # LOCATION: longest gazetteer phrase (up to 3 tokens)
        for L in (3, 2, 1):
            j = i + L - 1
            if j < n and all(self._only_space(text, toks[k].end, toks[k + 1].start)
                             for k in range(i, j)):
                phrase = re.sub(r"\s+", " ", text[toks[i].start:toks[j].end])
                if phrase in config.LOCATIONS:
                    return "LOCATION", j
        w = toks[i].text
        # PERSON: first name (+ optional known surname) or standalone surname
        if w in config.FIRST_NAMES:
            if (i + 1 < n and toks[i + 1].text in config.SURNAMES
                    and self._only_space(text, toks[i].end, toks[i + 1].start)):
                return "PERSON", i + 1
            return "PERSON", i
        if w in config.SURNAMES:
            return "PERSON", i
        if w.lower() in config.GENDERED_PRONOUNS:
            return "PRONOUN", i
        return None
