"""Rebuild the bundled CL term table from the OLS4 copy of the Cell Ontology."""

from __future__ import annotations

import json
from datetime import date
from importlib import resources
from pathlib import Path

from .ols import OLSClient
from .ontology import META_FILE, TERMS_FILE, write_tsv


def build_cl_table(out_dir: str | Path | None = None, ols: OLSClient | None = None) -> Path:
    """Download all non-obsolete CL classes from OLS4 and write ``cl_terms.tsv`` + metadata.

    Definitions come from IAO:0000115 only, so the long ChatGPT-generated
    "extended descriptions" in CL are not included. By default the files are
    written into the ``cl_mapper/data`` package folder.
    """
    ols = ols or OLSClient(timeout=120)
    out = Path(out_dir) if out_dir else Path(str(resources.files("cl_mapper.data")))
    out.mkdir(parents=True, exist_ok=True)
    version = ols.ontology_version("cl")
    terms = list(ols.iter_terms("cl"))
    if not terms:
        raise RuntimeError("OLS returned no CL terms")
    write_tsv(terms, out / TERMS_FILE)
    meta = {
        "cl_version": version,
        "source": f"{ols.base_url}/v2/ontologies/cl/classes",
        "built": date.today().isoformat(),
        "n_terms": len(terms),
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out / TERMS_FILE
