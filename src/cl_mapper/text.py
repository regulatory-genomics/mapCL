"""Label normalisation shared by user input and the direct-match index."""

from __future__ import annotations

import re
import unicodedata

UNKNOWN_LABELS = frozenset({
    "", "unk", "unknown", "na", "n a", "nan", "none", "null", "undefined",
    "unannotated", "not annotated", "unassigned", "not assigned",
})


def clean_label(label: object) -> str:
    """Remove technical noise from a user label before matching or searching.

    Drops cluster indices ("16_T cells", "Endothelial 3"), the word "like",
    quotes and other punctuation, and turns "cells" into "cell".
    Keeps "+", "-" and "/" because they carry marker information (CD4+, CD16-).
    """
    s = unicodedata.normalize("NFKC", str(label)).strip()
    s = re.sub(r"^\d+\s*[_.:]\s*", "", s)
    s = re.sub(r"[_.]\d+$", "", s)
    s = s.replace("_", " ")
    s = re.sub(r"[-\s]?\blike\b", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcells\b", "cell", s, flags=re.IGNORECASE)
    s = re.sub(r"[^\w\s+\-/]", " ", s)
    s = re.sub(r"\s+\d+$", "", s.strip())
    return re.sub(r"\s+", " ", s).strip(" -/")


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
    s = unicodedata.normalize("NFKC", text).lower().replace("_", " ")
    s = re.sub(r"[^\w\s+]", " ", s)
    return " ".join(sorted(_singular(t) for t in s.split()))
