from cl_mapper.ontology import CellOntology, Term, load_bundled, write_tsv


def test_direct_match_label_and_synonym(ontology):
    assert ontology.direct_match("T cells").id == "CL:0000084"
    assert ontology.direct_match("t lymphocyte").id == "CL:0000084"
    assert ontology.direct_match("NK cell").id == "CL:0000623"
    assert ontology.direct_match("Treg").id == "CL:0000815"


def test_ambiguous_synonym_is_not_matched(ontology):
    assert ontology.direct_match("dup") is None


def test_label_wins_over_synonym():
    terms = [Term("CL:1", "alpha cell"), Term("CL:2", "other", exact_synonyms=("alpha cell",))]
    onto = CellOntology({t.id: t for t in terms})
    assert onto.direct_match("alpha cell").id == "CL:1"


def test_related_synonyms_are_not_used_for_direct_match():
    onto = CellOntology({"CL:1": Term("CL:1", "microglial cell", other_synonyms=("microglia",))})
    assert onto.direct_match("microglia") is None


def test_get_normalises_ids(ontology):
    assert ontology.get("cl_0000084").label == "T cell"
    assert ontology.get("CL:0000000") is None


def test_tsv_roundtrip(tmp_path):
    terms = [Term("CL:1", "a cell", "def", ("x", "y"), ("z",))]
    write_tsv(terms, tmp_path / "t.tsv")
    loaded = CellOntology.from_tsv(tmp_path / "t.tsv")
    assert loaded.terms["CL:1"] == terms[0]


def test_bundled_table_loads():
    onto = load_bundled()
    assert len(onto.terms) > 3000
    assert onto.version
    assert onto.direct_match("B cells").id == "CL:0000236"
    assert all(t.id.startswith("CL:") for t in onto.terms.values())
