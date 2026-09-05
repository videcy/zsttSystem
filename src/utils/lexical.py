"""Shared lexical tokenisation and BM25 scoring for retrieval reranking.

The corpus is small (a few thousand Chinese syllabus chunks), so document
frequencies are computed offline once per index build and reused at query
time.  Keeping the tokeniser in one place guarantees that the statistics
written by :mod:`src.data_processing.lexical_stats` and the scores computed by
:class:`src.online_service.chroma_retriever.ChromaRetriever` agree on what a
term is.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Mapping

_LATIN_TOKEN = re.compile(r"[a-z][a-z0-9]+")
_NON_CJK = re.compile(r"[^\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Split text into latin words plus Chinese character bigrams.

    Bigrams are emitted with repetition so that term frequency is meaningful
    for BM25; use :func:`term_set` when only the vocabulary is needed.
    """
    lowered = (text or "").lower()
    tokens = _LATIN_TOKEN.findall(lowered)
    chinese = _NON_CJK.sub("", lowered)
    tokens.extend(
        chinese[index : index + 2] for index in range(len(chinese) - 1)
    )
    return tokens


def term_set(text: str) -> set[str]:
    """Vocabulary of ``text`` under :func:`tokenize`."""
    return set(tokenize(text))


def overlap_score(query_terms: set[str], document_terms: set[str]) -> float:
    """Legacy lexical score: share of query terms present in the document."""
    if not query_terms:
        return 0.0
    return len(query_terms & document_terms) / len(query_terms)


def idf(document_frequency: float, document_count: float) -> float:
    """Robertson/Sparck-Jones IDF with the usual +0.5 smoothing.

    Clamped at zero so that terms occurring in almost every document (``课程``,
    ``信息``, ``管理`` in this corpus) contribute nothing instead of a negative
    score.
    """
    if document_count <= 0:
        return 0.0
    numerator = document_count - document_frequency + 0.5
    denominator = document_frequency + 0.5
    return max(0.0, math.log(1.0 + numerator / denominator))


class BM25Scorer:
    """Okapi BM25 over precomputed corpus statistics.

    ``stats`` is the payload written by
    :func:`src.data_processing.lexical_stats.build_lexical_stats`.
    """

    def __init__(
        self,
        stats: Mapping[str, object],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.document_count = float(stats.get("document_count", 0) or 0)
        self.average_length = float(stats.get("average_length", 0.0) or 0.0)
        raw_frequencies = stats.get("document_frequency") or {}
        self.document_frequency: dict[str, float] = {
            str(term): float(count)
            for term, count in dict(raw_frequencies).items()
        }
        self._idf_cache: dict[str, float] = {}

    @property
    def usable(self) -> bool:
        """Whether the statistics are complete enough to score with."""
        return (
            self.document_count > 0
            and self.average_length > 0
            and bool(self.document_frequency)
        )

    def term_idf(self, term: str) -> float:
        cached = self._idf_cache.get(term)
        if cached is None:
            cached = idf(
                self.document_frequency.get(term, 0.0),
                self.document_count,
            )
            self._idf_cache[term] = cached
        return cached

    def score(self, query_tokens: Iterable[str], document_tokens: Iterable[str]) -> float:
        """Raw BM25 score of one document against one query."""
        document = list(document_tokens)
        if not document:
            return 0.0
        frequencies = Counter(document)
        length_norm = self.k1 * (
            1.0 - self.b + self.b * len(document) / self.average_length
        )
        total = 0.0
        for term in dict.fromkeys(query_tokens):
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            total += self.term_idf(term) * (
                frequency * (self.k1 + 1.0) / (frequency + length_norm)
            )
        return total

    def normalized_score(
        self,
        query_tokens: Iterable[str],
        document_tokens: Iterable[str],
    ) -> float:
        """BM25 mapped into ``[0, 1)`` so it can be linearly blended.

        The upper bound of a raw BM25 score depends on the query, so it is
        divided by the score a perfectly matching document would receive
        (every query term present with saturated term frequency).  Queries
        whose terms are all corpus-wide stopwords score 0.
        """
        query = list(dict.fromkeys(query_tokens))
        if not query:
            return 0.0
        ceiling = sum(
            self.term_idf(term) * (self.k1 + 1.0) for term in query
        )
        if ceiling <= 0.0:
            return 0.0
        return self.score(query, document_tokens) / ceiling
