"""Bundled Cell Ontology term table and the direct-match index built from it."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from importlib import resources
from pathlib import Path

from .text import GENERIC_TOKENS, QUALIFIERS, match_key, tokens

TERMS_FILE = "cl_terms.tsv"
META_FILE = "cl_terms.meta.json"
LIST_SEP = "|"


@dataclass(frozen=True)
class Term:
    id: str
    label: str
    definition: str = ""
    exact_synonyms: tuple[str, ...] = ()
    other_synonyms: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "definition": self.definition,
            "synonyms": list(self.exact_synonyms + self.other_synonyms),
        }


@dataclass
class CellOntology:
    """In-memory CL table with an exact-match index over labels and exact synonyms.

    A key that points to more than one term (for example a synonym shared by
    two classes) is left out of the index, so a direct match is never a guess.
    Labels win over synonyms when both produce the same key.
    """

    terms: dict[str, Term]
    version: str = ""
    _index: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        label_keys: dict[str, set[str]] = {}
        synonym_keys: dict[str, set[str]] = {}
        for term in self.terms.values():
            label_keys.setdefault(match_key(term.label), set()).add(term.id)
            for syn in term.exact_synonyms:
                synonym_keys.setdefault(match_key(syn), set()).add(term.id)
        index = {k: next(iter(ids)) for k, ids in synonym_keys.items() if len(ids) == 1}
        for key, ids in label_keys.items():
            if len(ids) == 1:
                index[key] = next(iter(ids))
            else:
                index.pop(key, None)
        index.pop("", None)
        self._index = index

    def direct_match(self, label: str) -> Term | None:
        term_id = self._index.get(match_key(label))
        return self.terms[term_id] if term_id else None

    @cached_property
    def vocabulary(self) -> frozenset[str]:
        """Word tokens of CL labels and exact synonyms, used to tell names from codes."""
        words = set(GENERIC_TOKENS | QUALIFIERS)
        for term in self.terms.values():
            for name in (term.label, *term.exact_synonyms):
                words.update(tokens(name))
        return frozenset(words)

    def get(self, term_id: str) -> Term | None:
        return self.terms.get(term_id.strip().upper().replace("_", ":"))

    @classmethod
    def from_tsv(cls, path: str | Path, version: str = "") -> CellOntology:
        terms: dict[str, Term] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                terms[row["id"]] = Term(
                    id=row["id"],
                    label=row["label"],
                    definition=row["definition"],
                    exact_synonyms=_split(row["exact_synonyms"]),
                    other_synonyms=_split(row["other_synonyms"]),
                )
        return cls(terms=terms, version=version)


def _split(value: str) -> tuple[str, ...]:
    return tuple(v for v in value.split(LIST_SEP) if v)


def write_tsv(terms: list[Term], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(["id", "label", "definition", "exact_synonyms", "other_synonyms"])
        for t in sorted(terms, key=lambda t: t.id):
            writer.writerow([
                t.id, t.label, t.definition,
                LIST_SEP.join(t.exact_synonyms), LIST_SEP.join(t.other_synonyms),
            ])


def bundled_metadata() -> dict:
    meta = resources.files("cl_mapper.data").joinpath(META_FILE)
    return json.loads(meta.read_text(encoding="utf-8")) if meta.is_file() else {}


@lru_cache(maxsize=1)
def load_bundled() -> CellOntology:
    """Load the CL table shipped in ``cl_mapper/data`` (built by ``clmap build-db``)."""
    path = resources.files("cl_mapper.data").joinpath(TERMS_FILE)
    with resources.as_file(path) as p:
        return CellOntology.from_tsv(p, version=bundled_metadata().get("cl_version", ""))
