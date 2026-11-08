import pytest

from cl_mapper.text import (
    clean_label,
    is_unknown,
    match_key,
    taxon,
    without_leading_words,
    without_qualifiers,
)


@pytest.mark.parametrize("raw, expected", [
    ("35-B cell", "B cell"),
    ("5-HT neuron", "5-HT neuron"),
    ("ITL6_2_1", "ITL6"),
    ("HypertrophicChon", "Hypertrophic Chon"),
    ("Bcell", "B cell"),
    ("Alveolar Type 1 (AT1) Cell", "Alveolar Type 1 Cell"),
    ("16_Endothelial cells", "Endothelial cell"),
    ("001. B cells", "B cell"),
    ("Endothelial 3", "Endothelial"),
    ("Fibroblast_2", "Fibroblast"),
    ("Monocyte-like", "Monocyte"),
    ("NK-like cells", "NK cell"),
    ("CD4+ T cell", "CD4+ T cell"),
    ("CD16- NK cell", "CD16- NK cell"),
    ("T/NK cells", "T/NK cell"),
    ('"Treg"', "Treg"),
    ("  B   cell  ", "B cell"),
])
def test_clean_label(raw, expected):
    assert clean_label(raw) == expected


@pytest.mark.parametrize("label", ["unknown", "Unknown", "NA", "N/A", "nan", "", "  ", "unassigned"])
def test_is_unknown(label):
    assert is_unknown(clean_label(label))


@pytest.mark.parametrize("label", ["NK", "T cell", "unknown T cell"])
def test_is_not_unknown(label):
    assert not is_unknown(label)


def test_match_key_ignores_case_order_punctuation_plural():
    assert match_key("T cells") == match_key("t-cell") == match_key("cell T")
    assert match_key("CD4-positive, alpha-beta T cell") == match_key("cd4 positive alpha beta t cells")


def test_match_key_keeps_marker_sign():
    assert match_key("CD4+ T cell") != match_key("CD4 T cell")


def test_match_key_marker_sign_equals_spelled_out():
    assert match_key("CD16- NK cell") == match_key("CD16-negative NK cell")


def test_taxon():
    assert taxon("STR D1 MSN neuron (Primate)") == "human"
    assert taxon("astrocyte (Mmus)") == "mouse"
    assert taxon("neuron (sensu Nematoda)") == "other"
    assert taxon("B cell (sensu Vertebrata)") is None
    assert taxon("hepatocyte") is None


def test_shortened_queries():
    assert without_qualifiers("Fetal Astrocyte") == "Astrocyte"
    assert without_leading_words("Fetal Atrial Cardiomyocyte") == ["Atrial Cardiomyocyte",
                                                                   "Cardiomyocyte"]


def test_match_key_keeps_latin_endings():
    assert match_key("plexus") == "plexus"
    assert match_key("glass") == "glass"
