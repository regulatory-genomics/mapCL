"""The mapping pipeline.

For every distinct (label, tissue, species, cell state) query:

1. Clean the label; placeholders such as "unknown" or "NA" stop here.
2. Direct match against CL labels and exact synonyms (bundled table).
3. OLS4 search in CL, then choose a term.
   Without an LLM: accept a CL term that contains every word of a query.
   With an LLM: the LLM scores the candidates; the best one is accepted at
   ``min_score`` or above.
   Optional LLM expansion (the LLM proposes the standard name) runs before the
   search for abbreviations and codes ("EC", "OGC_2"), and for full names only
   after every candidate was rejected; the search is then repeated.
4. With an LLM, if no CL term is accepted, NCIT/BTO are searched and scored.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from . import llm as llm_mod
from .ols import Candidate, OLSClient
from .ontology import CellOntology, Term, load_bundled
from .text import (
    QUALIFIERS,
    clean_label,
    content_tokens,
    is_abbreviation,
    is_short_code,
    is_unknown,
    match_key,
    normalize_species,
    taxon,
    tokens,
    without_leading_words,
    without_qualifiers,
)

AMBIGUOUS_HINT = ("If the label is an abbreviation with several plausible meanings and the "
                  "context does not decide between them, score at most 5.")
RESOLVED_HINT = ("The label was interpreted as '{expanded}', which is the CL term '{term}'. "
                 "Judge the candidates against this interpretation; do not lower the score "
                 "only because the label is an abbreviation.")
_TAXON_TOKENS = frozenset({"mmus", "primate", "human", "mouse", "homo", "sapien", "sapiens", "sensu"})

logger = logging.getLogger(__name__)

METHODS = {
    "direct": "cleaned label (or label + 'cell') equals a CL label or exact synonym",
    "expanded_direct": "LLM-expanded name equals a CL label or exact synonym (no assessment)",
    "lexical": "CL term whose label or a synonym has all words of the label or a shortened "
               "form of it, e.g. without 'fetal' (no LLM; often broader; see note)",
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
        """Map queries in parallel; output order follows input.

        Labels that clean to the same text in the same context ("Fetal Astrocyte 1",
        "Fetal Astrocyte 2") are mapped once.
        """
        items = [q if isinstance(q, Query) else Query(label=str(q)) for q in queries]
        first: dict[tuple, Query] = {}
        for q in items:
            first.setdefault(_key(q), q)
        unique = list(first.values())
        mapped = RunnableLambda(self._map_safe).batch(
            unique, config={"max_concurrency": self.max_concurrency})
        by_key = {_key(q): r for q, r in zip(unique, mapped, strict=True)}
        return [replace(by_key[_key(q)], input_label=q.label, cleaned_label=clean_label(q.label))
                for q in items]

    def map_one(self, query: Query | str) -> MappingResult:
        """Map a single query. OLS and LLM errors are raised, not caught."""
        q = query if isinstance(query, Query) else Query(label=str(query))
        cleaned = clean_label(q.label)
        res = MappingResult(input_label=q.label, cleaned_label=cleaned,
                            tissue=q.tissue, species=q.species, cell_state=q.cell_state)
        if is_unknown(cleaned):
            res.method = "unknown"
            return res
        # A short abbreviation can be the synonym of an unrelated term ("IC" is
        # "inferior cardiac neuron" in CL), so with an LLM it is only a candidate.
        if (term := self._direct(cleaned)) and not (self._assess and is_abbreviation(cleaned)):
            if is_abbreviation(cleaned):
                res.note = "abbreviation matched a CL synonym; check it"
            return _accept(res, _from_term(term), "direct")

        # Codes ("EC", "OGC_2", "ITL4") are expanded before searching. Full names
        # are searched and scored first and expanded only if every candidate fails.
        code = is_short_code(cleaned, self.ontology.vocabulary)
        base = _shortened(q.label, cleaned)
        if q.tissue and self._assess is not None:
            base.append(f"{q.tissue} {cleaned}")
        tried: set[str] = set()
        expanded: list[str] = []
        hint = AMBIGUOUS_HINT
        if code and self._expand is not None:
            expanded, hint = self._expansion(q, res)
        queries = _dedupe([*expanded, *base])
        if self._cl_round(q, res, queries, expanded, hint, tried):
            return res
        if not code and self._expand is not None:
            expanded, hint = self._expansion(q, res)
            queries = _dedupe([*expanded, *base])
            if self._cl_round(q, res, queries, expanded, hint, tried):
                return res

        if self._assess is not None and self.fallback_ontologies:
            fallback = self._candidates(q, queries[:3], self.fallback_ontologies)
            self._assess_step(q, res, fallback, "llm_fallback", hint)
        return res

    def _expansion(self, q: Query, res: MappingResult) -> tuple[list[str], str]:
        """LLM expansion: search names and the assessment hint."""
        exp = self._expand.invoke({"label": q.label, "context": self._context(q)})
        res.expanded_label = exp.cell_type
        names = _expanded_forms([exp.cell_type, *exp.synonyms[:2]])
        # An abbreviation whose expansion is itself a CL term is not ambiguous
        # any more, so the assessor must not cap its score.
        name = clean_label(exp.cell_type)
        if resolved := self._direct(name) or self._direct(without_qualifiers(name)):
            return names, RESOLVED_HINT.format(expanded=exp.cell_type, term=resolved.label)
        return names, AMBIGUOUS_HINT

    def _cl_round(self, q: Query, res: MappingResult, queries: list[str], expanded: list[str],
                  hint: str, tried: set[str]) -> bool:
        """Search CL with ``queries`` and choose a term; True if one is accepted.

        Candidates already rejected in an earlier round are not scored again.
        """
        if self._assess is None:
            return bool(self._lexical(res, q, queries, set(expanded)).mapped_id)
        cands = self._candidates(q, queries, ("cl",))
        if not cands or {c.id for c in cands} <= tried:
            return False
        tried.update(c.id for c in cands)
        return self._assess_step(q, res, cands, "llm", hint)

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

    def _direct(self, name: str) -> Term | None:
        """Direct match of ``name``, else of ``name + " cell"`` ("Plasma" -> plasma cell)."""
        term = self.ontology.direct_match(name)
        if term is None and "cell" not in tokens(name):
            term = self.ontology.direct_match(f"{name} cell")
        return term

    def _search(self, query: str, ontologies: Sequence[str], species: str | None) -> list[Candidate]:
        """OLS search; drops terms restricted to another species or to non-vertebrates."""
        if not any(len(t) > 1 for t in content_tokens(query)):
            return []  # single letters ("B") are slow and useless OLS queries
        hits = self.ols.search(query, ontologies, rows=self.top_k)
        keep = []
        for h in map(self._enrich, hits):
            tx = taxon(h.label) if h.ontology == "CL" else None
            if tx == "other" or (species and tx and tx != species):
                continue
            keep.append(h)
        return keep

    def _lexical(self, res: MappingResult, q: Query, queries: Sequence[str],
                 expanded: set[str]) -> MappingResult:
        """No LLM: accept a term only if it contains every word of a query.

        Queries are tried in order (expanded names, the label, shortened forms);
        among passing hits the one with the fewest extra words wins, so
        "B Memory" gives "memory B cell", not a narrow memory B subtype.
        """
        species = normalize_species(q.species)
        for query in queries:
            if term := self._direct(query):
                method = "expanded_direct" if query in expanded else "lexical"
                res.note = f"matched '{query}'"
                return _accept(res, _from_term(term), method)
            want = content_tokens(query)
            best, best_extra = None, None
            for h in self._search(query, ("cl",), species):
                extra = _extra_words(want, (h.label, *h.synonyms))
                if extra is not None and (best_extra is None or extra < best_extra):
                    best, best_extra = h, extra
            if best is not None:
                res.note = f"all words of '{query}' found in the term"
                if is_abbreviation(query):
                    res.note = f"abbreviation '{query}' is ambiguous without an LLM; check it"
                return _accept(res, best, "lexical")
        return res

    def _candidates(self, q: Query, names: Sequence[str], ontologies: Sequence[str]) -> list[Candidate]:
        """Direct CL matches of each name first, then OLS hits interleaved across queries."""
        species = normalize_species(q.species)
        merged: dict[str, Candidate] = {}
        if "cl" in ontologies:
            for name in names:
                if term := self._direct(name):
                    merged.setdefault(term.id, _from_term(term))
        hit_lists = [self._search(n, ontologies, species) for n in names]
        for rank in range(self.top_k):
            for hits in hit_lists:
                if rank < len(hits) and hits[rank].id not in merged:
                    merged[hits[rank].id] = hits[rank]
        return list(merged.values())[: self.max_candidates]

    def _assess_step(self, q: Query, res: MappingResult, cands: list[Candidate], method: str,
                     hint: str) -> bool:
        """Let the LLM pick a candidate; record score and reason; True if accepted."""
        if not cands:
            return False
        a = self._assess.invoke({"label": q.label, "context": self._context(q), "hint": hint,
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


def _key(q: Query) -> tuple:
    return match_key(clean_label(q.label)), q.tissue, q.species, q.cell_state


def _expanded_forms(names: Iterable[str]) -> list[str]:
    """Fix 1: search shorter forms of the LLM's names too, as OLS needs every word.

    "fetal ventricular cardiac muscle cell" adds "ventricular cardiac muscle cell";
    "subicular pyramidal neuron" adds "pyramidal neuron". A form must keep two
    content words, so "chondrocyte progenitor cell" does not add "progenitor cell".
    """
    out: list[str] = []
    for name in names:
        name = clean_label(name)
        plain = without_qualifiers(name)
        out += [name, plain]
        out += [v for v in without_leading_words(plain)[:1] if len(content_tokens(v)) >= 2]
    return [v for v in _dedupe(out) if content_tokens(v) - QUALIFIERS]


def _shortened(raw: str, cleaned: str) -> list[str]:
    """Search queries: the label, the part before a bracket ("Fibroblast (Peripheral
    Nerve)" -> "Fibroblast"), without qualifiers ("fetal"), then without the first
    1-2 words. Shortened forms that are one short fragment ("Endo", "PRO") are dropped.
    """
    head = clean_label(re.sub(r"\(.*$", "", raw)) or cleaned
    inner = re.search(r"\(([^)]*)\)", raw)
    moved = []  # "Fibroblast (Splenic)" -> "Splenic Fibroblast"
    if inner:
        for part in re.split(r"[,;/]", inner.group(1)):
            if (part := clean_label(part)) and part.lower() not in QUALIFIERS:
                moved.append(without_qualifiers(f"{part} {head}"))
    plain = without_qualifiers(head)
    shorter = [v for v in without_leading_words(plain)[:2] if " " in v or len(v) > 4]
    return [v for v in _dedupe([cleaned, *moved, head, plain, *shorter])
            if content_tokens(v) - QUALIFIERS]


def _extra_words(want: set[str], names: Iterable[str]) -> int | None:
    """Fewest words a name adds beyond ``want``; None if no name contains all of ``want``."""
    best = None
    for name in names:
        have = content_tokens(name) - _TAXON_TOKENS
        if want and want <= have:
            extra = len(have - want) + (2 if taxon(name) else 0)
            best = extra if best is None else min(best, extra)
    return best


def _accept(res: MappingResult, c: Candidate, method: str) -> MappingResult:
    res.mapped_id, res.mapped_label = c.id, c.label
    res.mapped_ontology, res.method = c.ontology, method
    return res
