"""Checks against the real OLS4 API. Run with: CLMAP_LIVE=1 pytest tests/test_live.py"""

import os

import pytest

from cl_mapper.mapper import CellOntologyMapper, Query
from cl_mapper.ols import OLSClient

pytestmark = [
    pytest.mark.network,
    pytest.mark.skipif(not os.getenv("CLMAP_LIVE"), reason="set CLMAP_LIVE=1 to call OLS4"),
]


def test_search_returns_only_defined_cl_classes():
    hits = OLSClient().search("liver", ("cl",), rows=5)
    assert hits and all(h.id.startswith("CL:") for h in hits)


def test_search_handles_special_characters():
    hits = OLSClient().search('"B cell (CD19+)', ("cl",), rows=3)
    assert hits[0].id == "CL:0001201"


def test_fallback_ontologies():
    hits = OLSClient().search("cancer associated fibroblast", ("ncit", "bto"), rows=3)
    assert hits and {h.ontology for h in hits} <= {"NCIT", "BTO"}


def test_mapper_without_llm():
    res = CellOntologyMapper().map([Query("B cells"), Query("Liver endothelial")])
    assert res[0].method == "direct" and res[0].mapped_id == "CL:0000236"
    assert res[1].method == "lexical" and res[1].mapped_id.startswith("CL:")
