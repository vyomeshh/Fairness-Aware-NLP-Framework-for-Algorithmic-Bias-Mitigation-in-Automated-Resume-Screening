"""M4 - sentence encoder f_theta (Eq. 3.2).

SBERTEncoder  : pretrained Sentence-BERT (PRIMARY; frozen, so identical for both
                conditions by construction).
TfidfEncoder  : offline FALLBACK stand-in (lexical, not semantic). Used only so
                the pipeline can run without model downloads. Must NOT be used
                to report results for the research questions.
"""
import warnings
import numpy as np


class SBERTEncoder:
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", seed: int = 42):
        from sentence_transformers import SentenceTransformer
        import torch
        torch.manual_seed(seed)
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def prepare(self, texts):          # pretrained: nothing to fit
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(list(texts), convert_to_numpy=True,
                                            show_progress_bar=False))


class TfidfEncoder:
    name = "tfidf-fallback"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # token_pattern keeps bracketed neutral tokens as ordinary words
        self.vec = TfidfVectorizer(lowercase=True, token_pattern=r"[A-Za-z][A-Za-z\-]+")
        self._fitted = False

    def prepare(self, texts):
        """Fit ONCE on the union of both conditions' texts + job descriptions so
        the vocabulary is identical for baseline and masked runs."""
        self.vec.fit(list(texts))
        self._fitted = True
        return self

    def encode(self, texts):
        if not self._fitted:
            raise RuntimeError("TfidfEncoder.prepare(texts) must be called first")
        return self.vec.transform(list(texts)).toarray()


def get_encoder(kind: str = "auto", **kw):
    if kind in ("auto", "sbert"):
        try:
            return SBERTEncoder(**kw)
        except Exception as exc:
            if kind == "sbert":
                raise RuntimeError(f"Sentence-BERT unavailable: {exc}")
            warnings.warn("sentence-transformers/model not available - using the TF-IDF "
                          "FALLBACK encoder (lexical stand-in). Do not report "
                          "experimental results with it.")
    return TfidfEncoder()
