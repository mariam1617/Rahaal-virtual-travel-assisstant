"""
Retriever for the Tourist Help RAG: BM25 over small, topic-sized chunks.

Two numbers decide what we return:
  * BM25 score    -> ranking.
  * coverage      -> how much of the question (idf-weighted) the chunk actually answers.
                     Questions the knowledge base knows nothing about ("best pizza in Rome")
                     get low coverage everywhere, so we say "I don't have that" instead of
                     returning the nearest unrelated paragraph.

Only the chunks that are close to the best hit are returned (RELATIVE_CUTOFF), so a narrow
question yields 1-3 chunks, never the whole document.

Module: src/rag/kb.py
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Set

from src.rag.loader import Chunk, load_all_chunks
from src.rag.text import query_groups, tokenize

MIN_COVERAGE = 0.40        # below this the chunk does not really answer the question
RELATIVE_CUTOFF = 0.50     # keep hits scoring >= 50% of the best hit
K1, B = 1.5, 0.75

_DOMAIN_HINTS = {
    "visa": re.compile(r"\b(visa|visas|evisa|e-visa|passport|entry|immigration|voa)\b|تأشيرة|تاشيرة|فيزا|جواز", re.I),
    "transit": re.compile(
        r"\b(taxi|taxis|cab|uber|careem|metro|subway|train|trains|railway|bus|buses|airport|airports|flight|flights|fly|"
        r"transport|transportation|ferry|felucca|cruise|cruises|rental|rent|bicycle|bike|motorcycle|getting around|get around)\b"
        r"|مواصلات|مترو|قطار|مطار|تاكسي|اتوبيس", re.I),
    "advisory": re.compile(
        r"\b(safe|safety|dangerous|danger|unsafe|risk|terror\w*|crime|scam\w*|touts?|harass\w*|dress|clothes|clothing|"
        r"alcohol|drugs?|photograph\w*|drones?|lgbt\w*|gay|ramadan|law|laws|illegal|insurance|warning|advis\w*|"
        r"border|permit|permits|desert|protest\w*|pickpocket\w*|balloon\w*|diving|snorkel\w*|quad)\b"
        r"|امان|أمان|آمنة|امنة|امنه|خطر|لبس|ملابس|كحول|تصوير|رمضان|تامين|تأمين", re.I),
}


@dataclass
class Hit:
    chunk: Chunk
    score: float
    coverage: float


class KnowledgeBase:
    def __init__(self, chunks: Sequence[Chunk]):
        self.chunks: List[Chunk] = list(chunks)
        self._tokens: List[List[str]] = [tokenize(c.index_text, keep_stopwords=True) for c in self.chunks]
        self._tf: List[Counter] = [Counter(t) for t in self._tokens]
        self._sets: List[Set[str]] = [set(t) for t in self._tokens]
        self._len = [len(t) for t in self._tokens]
        self._avg_len = (sum(self._len) / len(self._len)) if self._len else 1.0
        df: Counter = Counter()
        for s in self._sets:
            df.update(s)
        n = max(len(self.chunks), 1)
        self._idf: Dict[str, float] = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self._idf_unseen = math.log(1 + n)      # a word no document contains is "rare" => heavy weight
        self.visa_by_nationality: Dict[str, Chunk] = {
            c.meta["nationality"].lower(): c for c in self.chunks
            if c.kind == "visa_row" and c.meta.get("visa_type") == "Tourism e-Visa"
        }

    # ------------------------------------------------------------------ helpers
    def idf(self, term: str) -> float:
        return self._idf.get(term, self._idf_unseen)

    def group_weight(self, group: List[str]) -> float:
        return max(self.idf(t) for t in group)

    def _bm25(self, i: int, groups: List[List[str]]) -> float:
        score = 0.0
        tf, dl = self._tf[i], self._len[i]
        for g in groups:
            best = 0.0
            for term in g:
                f = tf.get(term, 0)
                if not f:
                    continue
                s = self.idf(term) * f * (K1 + 1) / (f + K1 * (1 - B + B * dl / self._avg_len))
                best = max(best, s)
            score += best
        return score

    def _coverage_of(self, words: Set[str], groups: List[List[str]]) -> float:
        total = sum(self.group_weight(g) for g in groups) or 1.0
        got = 0.0
        for g in groups:
            matched = [self.idf(t) for t in g if t in words]
            if matched:
                got += max(matched)          # a synonym match counts for what the synonym is worth
        return got / total

    def coverage(self, i: int, groups: List[List[str]]) -> float:
        return self._coverage_of(self._sets[i], groups)

    def text_coverage(self, text: str, groups: List[List[str]]) -> float:
        """Same idea as coverage(), for a sentence/bullet of a chunk (used by the answer extractor)."""
        return self._coverage_of(set(tokenize(text, keep_stopwords=True)), groups)

    # ------------------------------------------------------------------ public API
    @staticmethod
    def infer_domains(query: str) -> List[str]:
        found = [d for d, rx in _DOMAIN_HINTS.items() if rx.search(query)]
        return found or ["visa", "advisory", "transit"]

    def search(
        self,
        query: str,
        domains: Optional[Iterable[str]] = None,
        top_k: int = 3,
        min_coverage: float = MIN_COVERAGE,
        relative_cutoff: float = RELATIVE_CUTOFF,
    ) -> List[Hit]:
        groups = query_groups(query)
        if not groups:
            return []
        allowed = set(domains) if domains else None
        scored: List[Hit] = []
        for i, chunk in enumerate(self.chunks):
            if allowed is not None and chunk.domain not in allowed:
                continue
            s = self._bm25(i, groups)
            if s <= 0:
                continue
            cov = self.coverage(i, groups)
            scored.append(Hit(chunk, s * (0.5 + cov), cov))
        scored = [h for h in scored if h.coverage >= min_coverage]
        scored.sort(key=lambda h: h.score, reverse=True)
        if not scored:
            return []
        floor = scored[0].score * relative_cutoff
        return [h for h in scored if h.score >= floor][:top_k]


@lru_cache(maxsize=1)
def get_knowledge_base() -> KnowledgeBase:
    return KnowledgeBase(load_all_chunks())
