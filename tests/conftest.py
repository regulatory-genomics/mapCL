"""Offline fakes for OLS and the chat model, plus a small CL table."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from langchain_core.runnables import RunnableLambda

from mapcl.ols import Candidate
from mapcl.ontology import CellOntology, Term

TERMS = [
    Term("CL:0000084", "T cell", "A lymphocyte with a T cell receptor.", ("T-cell", "T lymphocyte")),
    Term("CL:0000236", "B cell", "A lymphocyte of B lineage.", ("B lymphocyte",)),
    Term("CL:0000623", "natural killer cell", "A cytotoxic lymphocyte.", ("NK cell",)),
    Term("CL:0000115", "endothelial cell", "Lines blood vessels.", ("endotheliocyte",)),
    Term("CL:1000398", "endothelial cell of hepatic sinusoid", "Liver sinusoid endothelium.",
         (), ("liver sinusoidal endothelial cell",)),
    Term("CL:0000057", "fibroblast", "A connective tissue cell."),
    Term("CL:0002553", "fibroblast of lung", "A lung fibroblast.", ("lung fibroblast",)),
    Term("CL:9000010", "alveolar fibroblast (Mmus)", "Mouse-only term."),
    Term("CL:9000011", "alveolar adventitial fibroblast", "General term."),
    Term("CL:0000091", "Kupffer cell", "Liver resident macrophage.", ("hepatic macrophage",)),
    Term("CL:0000235", "macrophage", "A phagocytic myeloid cell."),
    Term("CL:0000815", "regulatory T cell", "Suppresses immune responses.", ("Treg",)),
    Term("CL:9000001", "shared synonym one", "", ("dup",)),
    Term("CL:9000002", "shared synonym two", "", ("dup",)),
]


@pytest.fixture
def ontology() -> CellOntology:
    return CellOntology({t.id: t for t in TERMS}, version="test")


class FakeOLS:
    """Returns canned hits keyed by lower-cased query; records every call."""

    def __init__(self, hits: dict[tuple[str, str], list[Candidate]] | None = None):
        self.hits = hits or {}
        self.calls: list[tuple[str, tuple[str, ...], int]] = []

    def search(self, query, ontologies=("cl",), rows=5):
        self.calls.append((query, tuple(ontologies), rows))
        out: list[Candidate] = []
        for onto in ontologies:
            out += self.hits.get((query.lower(), onto), [])
        return out[:rows]


def cl(term_id: str, label: str) -> Candidate:
    return Candidate(term_id, label, "CL")


class FakeLLM:
    """Stands in for a chat model: ``with_structured_output`` calls ``responder(schema, prompt)``."""

    def __init__(self, responder: Callable):
        self.responder = responder
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        def run(prompt_value):
            text = prompt_value.to_string()
            self.prompts.append(text)
            return self.responder(schema, text)
        return RunnableLambda(run)


@pytest.fixture
def fake_ols():
    return FakeOLS
