import pytest

from cl_mapper.llm import Assessment, Expansion
from cl_mapper.mapper import CellOntologyMapper, Query
from cl_mapper.ols import Candidate, OLSError

from .conftest import FakeLLM, FakeOLS, cl


def make(ontology, ols=None, llm=None, **kw):
    return CellOntologyMapper(llm, ontology=ontology, ols=ols or FakeOLS(), **kw)


def test_unknown_and_direct_need_no_network(ontology):
    ols = FakeOLS()
    res = make(ontology, ols).map(["unknown", "NA", "B cells", "16_T cells"])
    assert [r.method for r in res] == ["unknown", "unknown", "direct", "direct"]
    assert [r.mapped_id for r in res] == [None, None, "CL:0000236", "CL:0000084"]
    assert ols.calls == []


def test_no_llm_needs_all_query_words_in_term(ontology):
    # macrophage is OLS's top hit but lacks "liver"/"endothelial"; the sinusoid term has both
    ols = FakeOLS({("liver endothelial", "cl"): [cl("CL:0000235", "x"), cl("CL:1000398", "y")]})
    r = make(ontology, ols).map_one("Liver endothelial")
    assert (r.mapped_id, r.method) == ("CL:1000398", "lexical")
    assert r.mapped_label == "endothelial cell of hepatic sinusoid"  # bundled label, not OLS text


def test_no_llm_skips_terms_of_other_species(ontology):
    ols = FakeOLS({("alveolar fibroblast", "cl"): [cl("CL:9000010", "x")]})
    human = make(ontology, ols).map_one(Query("Alveolar fibroblast", species="human"))
    assert human.mapped_id == "CL:0000057"  # mouse term skipped -> broader "fibroblast"
    mouse = make(ontology, ols).map_one(Query("Alveolar fibroblast", species="mouse"))
    assert mouse.mapped_id == "CL:9000010"


def test_no_llm_prefers_general_over_species_specific_term(ontology):
    hits = [cl("CL:9000010", "x"), cl("CL:9000011", "y")]
    ols = FakeOLS({("alveolar fibroblast", "cl"): hits})
    assert make(ontology, ols).map_one("Alveolar fibroblast").mapped_id == "CL:9000011"


def test_no_llm_falls_back_to_shorter_query(ontology):
    r = make(ontology).map_one("Fetal Kupffer cells 3")
    assert (r.mapped_id, r.method) == ("CL:0000091", "lexical")
    assert "Kupffer" in r.note


def test_bracket_qualifier_moved_into_query(ontology):
    r = make(ontology).map_one("Fibroblast (Lung)")
    assert r.mapped_id == "CL:0002553"


def test_label_plus_cell_is_direct(ontology):
    assert make(ontology).map_one("Macrophage").method == "direct"
    assert make(ontology).map_one("Kupffer").mapped_id == "CL:0000091"


def test_no_hits_is_unmapped(ontology):
    r = make(ontology).map_one("weird cluster")
    assert (r.mapped_id, r.method) == (None, "unmapped")


def test_variants_of_one_label_are_mapped_once(ontology):
    ols = FakeOLS()
    res = make(ontology, ols).map(["Weird 1", "Weird 2", "weird_3"])
    assert [r.input_label for r in res] == ["Weird 1", "Weird 2", "weird_3"]
    assert len({c[0] for c in ols.calls}) == len(ols.calls)  # no repeated searches


def test_abbreviation_direct_hit_goes_to_llm(ontology):
    llm = FakeLLM(lambda schema, _: Assessment(best_index=1, score=3, reason="no"))
    r = make(ontology, FakeOLS(), llm, fallback_ontologies=()).map_one("Treg")
    assert (r.mapped_id, r.score) == (None, 3)  # LLM saw the candidate and rejected it
    assert make(ontology).map_one("Treg").method == "direct"  # no LLM: still direct, with a note


def test_llm_assessment_accepts_best_candidate(ontology):
    ols = FakeOLS({("kc", "cl"): [cl("CL:0000235", "macrophage"), cl("CL:0000091", "Kupffer cell")]})
    llm = FakeLLM(lambda schema, _: Assessment(best_index=2, score=9, reason="liver macrophage"))
    r = make(ontology, ols, llm).map_one(Query("KC", tissue="liver"))
    assert (r.mapped_id, r.method, r.score) == ("CL:0000091", "llm", 9)
    assert "Kupffer cell [CL:0000091]" in llm.prompts[0]
    assert "Tissue: liver" in llm.prompts[0]
    assert ("liver KC", ("cl",), 5) in ols.calls  # tissue-qualified query is searched too


def test_low_score_falls_back_to_ncit_bto(ontology):
    ols = FakeOLS({
        ("caf", "cl"): [cl("CL:0000235", "macrophage")],
        ("caf", "ncit"): [Candidate("NCIT:C168534", "Cancer-Associated Fibroblast", "NCIT")],
    })
    scores = iter([3, 9])
    llm = FakeLLM(lambda schema, _: Assessment(best_index=1, score=next(scores), reason="r"))
    r = make(ontology, ols, llm).map_one("CAF")
    assert (r.mapped_id, r.mapped_ontology, r.method, r.score) == (
        "NCIT:C168534", "NCIT", "llm_fallback", 9)


def test_low_score_everywhere_is_unmapped_with_score(ontology):
    ols = FakeOLS({("blob", "cl"): [cl("CL:0000235", "macrophage")]})
    llm = FakeLLM(lambda schema, _: Assessment(best_index=1, score=4, reason="unrelated"))
    r = make(ontology, ols, llm).map_one("blob")
    assert (r.mapped_id, r.method, r.score, r.note) == (None, "unmapped", 4, "unrelated")


def test_invalid_llm_index_is_rejected(ontology):
    ols = FakeOLS({("blob", "cl"): [cl("CL:0000235", "macrophage")]})
    llm = FakeLLM(lambda schema, _: Assessment(best_index=7, score=9, reason="r"))
    r = make(ontology, ols, llm, fallback_ontologies=()).map_one("blob")
    assert r.mapped_id is None and "invalid" in r.note


def test_expansion_feeds_direct_match_and_search(ontology):
    def respond(schema, prompt):
        if schema is Expansion:
            return Expansion(cell_type="Kupffer cell", synonyms=["hepatic macrophage"])
        return Assessment(best_index=1, score=10, reason="exact")
    ols = FakeOLS()
    r = make(ontology, ols, FakeLLM(respond), expand=True).map_one(Query("KC", tissue="liver"))
    assert (r.mapped_id, r.method, r.expanded_label) == ("CL:0000091", "llm", "Kupffer cell")
    assert [c[0] for c in ols.calls][:2] == ["Kupffer cell", "hepatic macrophage"]


def test_expansion_without_assessment_uses_expanded_direct(ontology):
    llm = FakeLLM(lambda schema, _: Expansion(cell_type="natural killer cell", synonyms=[]))
    r = make(ontology, FakeOLS(), llm, expand=True, assess=False).map_one("NKs")
    assert (r.mapped_id, r.method) == ("CL:0000623", "expanded_direct")


def test_expand_requires_llm(ontology):
    with pytest.raises(ValueError):
        make(ontology, expand=True)


def test_errors_are_isolated_per_label(ontology):
    class BrokenOLS(FakeOLS):
        def search(self, query, ontologies=("cl",), rows=5):
            raise OLSError("down")
    res = make(ontology, BrokenOLS()).map(["B cell", "mystery"])
    assert [r.method for r in res] == ["direct", "error"]
    assert "down" in res[1].note


def test_duplicates_are_mapped_once_and_order_kept(ontology):
    ols = FakeOLS({("x1", "cl"): [cl("CL:0000235", "m")]})
    res = make(ontology, ols).map(["x1", "B cell", "x1"])
    assert [r.input_label for r in res] == ["x1", "B cell", "x1"]
    assert len(ols.calls) == 1
    assert res[0] is not res[2]


def test_same_label_different_tissue_is_mapped_separately(ontology):
    ols = FakeOLS({("endo", "cl"): [cl("CL:0000115", "e")]})
    make(ontology, ols).map([Query("endo", tissue="liver"), Query("endo", tissue="lung")])
    assert len(ols.calls) == 2
