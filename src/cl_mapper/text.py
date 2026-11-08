"""Label normalisation shared by user input and the direct-match index."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Set as AbstractSet

UNKNOWN_LABELS = frozenset({
    "", "unk", "unknown", "unknown cell", "unknown cell type", "na", "n a", "nan", "none",
    "null", "undefined", "unannotated", "not annotated", "unassigned", "not assigned",
    "unclassified", "unclassified cell", "unidentified", "unidentified cell",
})
GENERIC_TOKENS = frozenset({"cell", "type", "of", "and", "the", "in", "positive", "negative"})
# Qualifiers that CL rarely encodes; dropping them gives a correct but broader term.
QUALIFIERS = frozenset({
    "fetal", "foetal", "embryonic", "adult", "immature", "cycling", "proliferating",
    "dividing", "activated", "resting", "general", "normal", "healthy",
})
_SPECIES = {"human": "human", "homo sapiens": "human", "hs": "human", "hsa": "human",
            "mouse": "mouse", "mus musculus": "mouse", "mm": "mouse", "mmu": "mouse"}


def clean_label(label: object) -> str:
    """Remove technical noise from a user label before matching or searching.

    - cluster indices: "16_T cells", "35-B cell", "Endothelial 3", "ITL6_2_1"
    - bracketed abbreviations: "Alveolar Type 1 (AT1) Cell" -> "Alveolar Type 1 Cell"
    - glued words: "HypertrophicChon" -> "Hypertrophic Chon", "Bcells" -> "B cell"
    - the word "like", quotes and other punctuation; "cells" -> "cell"
    Keeps "+", "-" and "/" inside words because they carry marker information.
    The LLM always sees the original label, so nothing here is lost for it.
    """
    s = unicodedata.normalize("NFKC", str(label)).strip()
    s = re.sub(r"\(\s*[A-Z0-9]{1,5}\s*\)", " ", s)  # "(AT1)" repeats the words before it
    s = re.sub(r"^\d+\s*(?:[_.:]|-(?![A-Z]{2,}\b))\s*", "", s)  # keeps "5-HT neuron"
    s = re.sub(r"(?:[_.]\d+)+$", "", s.strip())
    s = s.replace("_", " ")
    s = re.sub(r"(?<=[a-z]{3})(?=[A-Z])", " ", s)
    s = re.sub(r"\b([A-Z][A-Za-z0-9+]{0,4})(cells?)\b", r"\1 \2", s)
    s = re.sub(r"[-\s]?\blike\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcells\b", "cell", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s+\-/]", " ", s)
    s = re.sub(r"(\s+\d+)+$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip(" -/")


def is_abbreviation(label: str) -> bool:
    """Single short token such as "IC", "EC", "COP": a direct hit may be a different cell type."""
    return " " not in label and len(label) <= 4


_CODE_PART = re.compile(r"[A-Z]{2,}[A-Za-z0-9]*|[A-Za-z]+\d[A-Za-z0-9]*")
_MARKER = re.compile(r"CD\d+[A-Za-z]?|[A-Za-z0-9]+[+-]")


def is_short_code(label: str, vocabulary: AbstractSet[str]) -> bool:
    """Abbreviation or cluster code ("EC", "OGC", "ITL4", "Myofib PRO", "EN-IT-Immature").

    A full name is made only of words that occur in CL names ("Fetal Hepatic
    Endothelial"); capitalised codes and words with digits make it a code.
    Numbers and markers ("CD4", "NPY+") do not count.
    """
    if is_abbreviation(label):
        return True
    for part in re.split(r"[\s/.]+|-(?=\w)", label):
        if not part or part.isdigit() or _MARKER.fullmatch(part):
            continue
        if _CODE_PART.fullmatch(part) or any(t not in vocabulary for t in tokens(part)):
            return True
    return False


def without_leading_words(label: str) -> list[str]:
    """Shorter search queries, e.g. "Fetal Atrial Cardiomyocyte" -> ["Atrial Cardiomyocyte", "Cardiomyocyte"].

    OLS requires every query word to match, so one unknown word gives no hits.
    Suffixes made only of generic words ("cell") are skipped.
    """
    words = label.split()
    out = [" ".join(words[i:]) for i in range(1, len(words))]
    return [v for v in out if set(tokens(v)) - GENERIC_TOKENS]


def without_qualifiers(label: str) -> str:
    """Drop words like "fetal" or "cycling": "Fetal Astrocyte 3" -> "Astrocyte"."""
    return " ".join(w for w in label.split() if w.lower() not in QUALIFIERS)


def content_tokens(text: str) -> set[str]:
    return set(tokens(text)) - GENERIC_TOKENS


def taxon(label: str) -> str | None:
    """Species a CL label is restricted to: "mouse", "human", "other", or None (general)."""
    low = label.lower()
    if low.endswith(("(mmus)", "(sensu mus)", ", mouse")):
        return "mouse"
    if low.endswith(("(primate)", "(homo sapiens)", ", human")):
        return "human"
    if re.search(r"\(sensu (?!vertebrata)[^)]*\)$", low):
        return "other"
    return None


def is_unknown(label: str) -> bool:
    """True for placeholder labels such as "unknown", "NA" or an empty string."""
    return re.sub(r"[^\w]+", " ", label.lower()).strip() in UNKNOWN_LABELS


def _singular(token: str) -> str:
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def match_key(text: str) -> str:
    """Key for exact lookup that ignores case, punctuation, plurals and word order.

    The same function is applied to ontology names and to user labels, so
    "T cells", "t-cell" and "cell T" all give the key of "T cell".
    """
    return " ".join(sorted(tokens(text)))


def tokens(text: str) -> list[str]:
    """Lower-case singular word tokens; marker signs become words ("CD4+" -> "cd4 positive")."""
    s = unicodedata.normalize("NFKC", text).lower().replace("_", " ")
    s = re.sub(r"(\w)\+", r"\1 positive ", s)
    s = re.sub(r"(\w)-(?=\s|$|,|/)", r"\1 negative ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return [_singular(t) for t in s.split()]


def normalize_species(species: str) -> str | None:
    return _SPECIES.get(species.strip().lower()) if species else None
