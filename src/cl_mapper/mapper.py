"""The mapping pipeline.

For every distinct (label, tissue, species, cell state) query:

1. Clean the label; placeholders such as "unknown" or "NA" stop here.
2. Direct match against CL labels and exact synonyms (bundled table).
3. Optional LLM expansion: the LLM proposes the standard name, which is tried
   as a direct match and as an extra search query.
4. OLS4 search in CL.
5. Without an LLM: accept the top CL hit.
   With an LLM: the LLM scores the candidates; the best one is accepted at
   ``min_score`` or above, otherwise NCIT/BTO are searched and scored the same way.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from . import llm as llm_mod
from .ols import Candidate, OLSClient
from .ontology import CellOntology, Term, load_bundled
from .text import clean_label, is_unknown

logger = logging.getLogger(__name__)

METHODS = {
    "direct": "exact match of the cleaned label to a CL label or exact synonym",
    "expanded_direct": "exact match of the LLM-expanded name (no LLM assessment)",
    "ols_top_hit": "first CL hit from OLS search (no LLM assessment)",
    "llm": "CL candidate accepted by the LLM assessment",
    "llm_fallback": "NCIT/BTO candidate accepted by the LLM assessment",
    "unknown": "placeholder label such as 'unknown' or 'NA'",
    "unmapped": "no acceptable candidate",
    "error": "OLS or LLM call failed; see note",
}


@dataclass(frozen=True)
class Query:
    label: str
    tissue: str = ""
    species: str = ""
    cell_state: str = ""


@dataclass
class MappingResult:
    input_label: str
    cleaned_label: str
    tissue: str = ""
    species: str = ""
    cell_state: str = ""
    mapped_id: str | None = None
    mapped_label: str | None = None
    mapped_ontology: str | None = None
    method: str = "unmapped"
    score: int | None = None
    expanded_label: str | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CellOntologyMapper:
    """Map cell type labels to Cell Ontology terms.

    Args:
        llm: LangChain chat model or "provider:model" string. ``None`` runs
            without an LLM (direct match + OLS top hit).
        expand: ask the LLM for the standard name before searching.
        assess: let the LLM score candidates and reject weak ones.
        study_context: free text added to every LLM prompt, e.g. "human PBMC, COVID-19".
        min_score: lowest LLM score (1-10) accepted as a match.
        top_k: OLS hits requested per search query.
        max_candidates: candidates shown to the LLM after merging queries.
        fallback_ontologies: searched when no CL candidate passes the assessment.
        max_concurrency: queries processed in parallel.
        ontology, ols: override the bundled CL table or the OLS client (tests).
    """

    def __init__(
        self,
        llm: BaseChatModel | str | None = None,
        *,
        expand: bool = False,
        assess: bool = True,
        study_context: str | None = None,
        min_score: int = 6,
        top_k: int = 5,
        max_candidates: int = 8,
        fallback_ontologies: Sequence[str] = ("ncit", "bto"),
        max_concurrency: int = 4,
        ontology: CellOntology | None = None,
        ols: OLSClient | None = None,
    ):
        if isinstance(llm, str):
            llm = llm_mod.make_chat_model(llm)
        if expand and llm is None:
            raise ValueError("expand=True needs an llm")
        self.llm = llm
        self.study_context = study_context or ""
        self.min_score = min_score
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.fallback_ontologies = tuple(fallback_ontologies)
        self.max_concurrency = max_concurrency
        self.ontology = ontology or load_bundled()
        self.ols = ols or OLSClient()
        self._expand = llm_mod.expansion_chain(llm) if llm is not None and expand else None
        self._assess = llm_mod.assessment_chain(llm) if llm is not None and assess else None

    def map(self, queries: Iterable[Query | str]) -> list[MappingResult]:
        """Map queries in parallel. Duplicates are mapped once; output order follows input."""
        items = [q if isinstance(q, Query) else Query(label=str(q)) for q in queries]
        unique = list(dict.fromkeys(items))
        mapped = RunnableLambda(self._map_safe).batch(
            unique, config={"max_concurrency": self.max_concurrency})
        by_query = dict(zip(unique, mapped, strict=True))
        return [replace(by_query[q]) for q in items]

    def map_one(self, query: Query | str) -> MappingResult:
        """Map a single query. OLS and LLM errors are raised, not caught."""
        q = query if isinstance(query, Query) else Query(label=str(query))
        cleaned = clean_label(q.label)
        res = MappingResult(input_label=q.label, cleaned_label=cleaned,
                            tissue=q.tissue, species=q.species, cell_state=q.cell_state)
        if is_unknown(cleaned):
            res.method = "unknown"
            return res
        if term := self.ontology.direct_match(cleaned):
            return _accept(res, _from_term(term), "direct")

        names = [cleaned]
        if self._expand is not None:
            exp = self._expand.invoke({"label": q.label, "context": self._context(q)})
            res.expanded_label = exp.cell_type
            names = _dedupe([exp.cell_type, *exp.synonyms[:2], cleaned])

        if self._assess is None:
            return self._top_hit(res, names)

        if q.tissue:
            names.append(f"{q.tissue} {cleaned}")
        if self._assess_step(q, res, self._candidates(names, ("cl",)), "llm"):
            return res
        if self.fallback_ontologies:
            fallback = self._candidates(names, self.fallback_ontologies)
            self._assess_step(q, res, fallback, "llm_fallback")
        return res

    def _map_safe(self, q: Query) -> MappingResult:
        try:
            return self.map_one(q)
        except Exception as exc:  # one failing label must not abort the batch
            logger.warning("Mapping failed for %r: %s", q.label, exc)
            return MappingResult(input_label=q.label, cleaned_label=clean_label(q.label),
                                 tissue=q.tissue, species=q.species, cell_state=q.cell_state,
                                 method="error", note=f"{type(exc).__name__}: {exc}")

    def _context(self, q: Query) -> str:
        return llm_mod.format_context(tissue=q.tissue, species=q.species,
                                      cell_state=q.cell_state, study_context=self.study_context)

    def _enrich(self, c: Candidate) -> Candidate:
        """Use the bundled CL definition and synonyms when the term is known."""
        term = self.ontology.get(c.id) if c.ontology == "CL" else None
        return _from_term(term) if term else c

    def _top_hit(self, res: MappingResult, names: Sequence[str]) -> MappingResult:
        """No assessment: first direct match of an expanded name, else first OLS CL hit."""
        for name in names:
            if term := self.ontology.direct_match(name):
                return _accept(res, _from_term(term), "expanded_direct")
            if hits := self.ols.search(name, ("cl",), rows=1):
                return _accept(res, self._enrich(hits[0]), "ols_top_hit")
        return res

    def _candidates(self, names: Sequence[str], ontologies: Sequence[str]) -> list[Candidate]:
        """Direct CL matches of each name first, then OLS hits interleaved across queries."""
        merged: dict[str, Candidate] = {}
        if "cl" in ontologies:
            for name in names:
                if term := self.ontology.direct_match(name):
                    merged.setdefault(term.id, _from_term(term))
        hit_lists = [self.ols.search(n, ontologies, rows=self.top_k) for n in names]
        for rank in range(self.top_k):
            for hits in hit_lists:
                if rank < len(hits) and hits[rank].id not in merged:
                    merged[hits[rank].id] = self._enrich(hits[rank])
        return list(merged.values())[: self.max_candidates]

    def _assess_step(self, q: Query, res: MappingResult, cands: list[Candidate], method: str) -> bool:
        """Let the LLM pick a candidate; record score and reason; True if accepted."""
        if not cands:
            return False
        a = self._assess.invoke({"label": q.label, "context": self._context(q),
                                 "candidates": llm_mod.format_candidates(cands)})
        if not (1 <= a.best_index <= len(cands) and 1 <= a.score <= 10):
            res.note = f"LLM returned invalid index/score ({a.best_index}, {a.score})"
            return False
        res.score, res.note = a.score, a.reason
        if a.score < self.min_score:
            return False
        _accept(res, cands[a.best_index - 1], method)
        return True


def _from_term(term: Term) -> Candidate:
    return Candidate(id=term.id, label=term.label, ontology="CL", definition=term.definition,
                     synonyms=term.exact_synonyms + term.other_synonyms)


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(s.strip() for s in items if s and s.strip()))


def _accept(res: MappingResult, c: Candidate, method: str) -> MappingResult:
    res.mapped_id, res.mapped_label = c.id, c.label
    res.mapped_ontology, res.method = c.ontology, method
    return res
