"""Client for the EBI Ontology Lookup Service (OLS4)."""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .ontology import Term

OLS_URL = "https://www.ebi.ac.uk/ols4/api"
_SEARCH_FIELDS = "label,obo_id,ontology_prefix,description,synonym,is_defining_ontology"
_SYN = "http://www.geneontology.org/formats/oboInOwl#has{}Synonym"
_DEF = "http://purl.obolibrary.org/obo/IAO_0000115"
_SOLR_SPECIAL = re.compile(r'["():{}\[\]^~*?\\!]')


class OLSError(RuntimeError):
    """OLS could not be reached or returned an unusable response."""


@dataclass(frozen=True)
class Candidate:
    id: str
    label: str
    ontology: str
    definition: str = ""
    synonyms: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "label": self.label, "ontology": self.ontology,
            "definition": self.definition, "synonyms": list(self.synonyms),
        }


class OLSClient:
    """Thread-safe OLS4 client with retries and an in-memory search cache."""

    def __init__(self, base_url: str = OLS_URL, timeout: float = 30.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        retry = Retry(total=retries, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504))
        self._session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))
        self._cache: dict[tuple, list[Candidate]] = {}
        self._lock = threading.Lock()

    def _get(self, path: str, params: dict) -> dict:
        try:
            resp = self._session.get(f"{self.base_url}{path}", params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise OLSError(f"OLS request {path} failed: {exc}") from exc

    def search(self, query: str, ontologies: Sequence[str] = ("cl",), rows: int = 5) -> list[Candidate]:
        """Return up to ``rows`` classes defined in ``ontologies`` that match ``query``.

        ``local=true`` and the ID-prefix check drop terms an ontology only
        imports (for example UBERON "liver" inside CL).
        """
        query = " ".join(_SOLR_SPECIAL.sub(" ", query).split())
        if not query:
            return []
        key = (query.lower(), tuple(ontologies), rows)
        with self._lock:
            if key in self._cache:
                return list(self._cache[key])
        data = self._get("/search", {
            "q": query, "ontology": ",".join(ontologies), "rows": rows * 2,
            "type": "class", "local": "true", "obsoletes": "false", "fieldList": _SEARCH_FIELDS,
        })
        prefixes = {o.upper() for o in ontologies}
        results: list[Candidate] = []
        for doc in data.get("response", {}).get("docs", []):
            obo_id = doc.get("obo_id") or ""
            prefix = obo_id.split(":", 1)[0]
            if prefix not in prefixes or doc.get("is_defining_ontology") is False:
                continue
            desc = doc.get("description") or []
            results.append(Candidate(
                id=obo_id,
                label=doc.get("label") or obo_id,
                ontology=prefix,
                definition=_shorten((desc[0] if isinstance(desc, list) and desc else desc) or ""),
                synonyms=tuple(doc.get("synonym") or ()),
            ))
            if len(results) == rows:
                break
        with self._lock:
            self._cache[key] = results
        return list(results)

    def ontology_version(self, ontology: str = "cl") -> str:
        cfg = self._get(f"/ontologies/{ontology}", {}).get("config", {})
        return cfg.get("version") or ""

    def iter_terms(self, ontology: str = "cl", page_size: int = 100) -> Iterator[Term]:
        """Yield every non-obsolete class defined in ``ontology`` (OLS4 v2 API)."""
        page, prefix = 0, ontology.upper() + ":"
        while True:
            data = self._get(f"/v2/ontologies/{ontology}/classes",
                             {"size": page_size, "page": page, "isDefiningOntology": "true"})
            for el in data.get("elements", []):
                curie = el.get("curie") or ""
                if el.get("isObsolete") or not curie.startswith(prefix):
                    continue
                yield Term(
                    id=curie,
                    label=_first(el.get("label")) or curie,
                    definition=_first(el.get(_DEF)),
                    exact_synonyms=_values(el.get(_SYN.format("Exact"))),
                    other_synonyms=tuple(v for kind in ("Related", "Broad", "Narrow")
                                         for v in _values(el.get(_SYN.format(kind)))),
                )
            page += 1
            if page >= data.get("totalPages", 0):
                return


def _value(item) -> str:
    if isinstance(item, dict):
        item = item.get("value", "")
    return " ".join(str(item).split())


def _values(raw) -> tuple[str, ...]:
    items = raw if isinstance(raw, list) else [raw] if raw else []
    seen: dict[str, None] = {}
    for item in items:
        if (v := _value(item)):
            seen.setdefault(v, None)
    return tuple(seen)


def _first(raw) -> str:
    vals = _values(raw)
    return vals[0] if vals else ""


def _shorten(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3].rsplit(" ", 1)[0] + "..."
