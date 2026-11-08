"""Table-level helpers: map labels from lists, DataFrames, TSV/CSV files or AnnData."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .mapper import CellOntologyMapper, Query

OUTPUT_COLUMNS = [
    "mapped_id", "mapped_label", "mapped_ontology", "method", "score",
    "expanded_label", "cleaned_label", "note",
]


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="," if path.suffix.lower() == ".csv" else "\t", index=False)
    return path


def _column(df: pd.DataFrame, name: str | None, what: str) -> list[str] | None:
    if name is None:
        return None
    if name not in df.columns:
        raise KeyError(f"{what} column {name!r} not found; columns are {list(df.columns)}")
    return df[name].astype(object).where(df[name].notna(), "").astype(str).str.strip().tolist()


def map_dataframe(
    df: pd.DataFrame,
    label_col: str,
    *,
    tissue_col: str | None = None,
    species_col: str | None = None,
    cell_state_col: str | None = None,
    mapper: CellOntologyMapper | None = None,
    **mapper_kwargs: Any,
) -> pd.DataFrame:
    """Return a copy of ``df`` with the mapping columns appended.

    Each distinct (label, tissue, species, cell state) combination is mapped
    once, so the same label in two tissues can map to two different terms.
    """
    mapper = mapper or CellOntologyMapper(**mapper_kwargs)
    labels = _column(df, label_col, "label")
    blank = [""] * len(df)
    tissue = _column(df, tissue_col, "tissue")
    species = _column(df, species_col, "species")
    state = _column(df, cell_state_col, "cell state")
    queries = [
        Query(label=lab, tissue=t, species=s, cell_state=c)
        for lab, t, s, c in zip(labels, blank if tissue is None else tissue,
                                blank if species is None else species,
                                blank if state is None else state, strict=True)
    ]
    results = mapper.map(queries)
    out = df.copy()
    for col in OUTPUT_COLUMNS:
        out[col] = [getattr(r, col) for r in results]
    out["score"] = out["score"].astype("Int64")
    return out


def map_labels(
    labels: Sequence[str] | str,
    *,
    tissue: str = "",
    species: str = "",
    cell_state: str = "",
    mapper: CellOntologyMapper | None = None,
    **mapper_kwargs: Any,
) -> pd.DataFrame:
    """Map a list of labels that share one tissue/species/cell state."""
    if isinstance(labels, str):
        labels = [labels]
    df = pd.DataFrame({"input_label": list(labels), "tissue": tissue,
                       "species": species, "cell_state": cell_state})
    return map_dataframe(df, "input_label", tissue_col="tissue", species_col="species",
                         cell_state_col="cell_state", mapper=mapper, **mapper_kwargs)


def map_adata(
    adata,
    label_col: str = "cell_type",
    *,
    tissue_col: str | None = None,
    species_col: str | None = None,
    cell_state_col: str | None = None,
    key_added: str = "cl",
    mapper: CellOntologyMapper | None = None,
    **mapper_kwargs: Any,
):
    """Add ``{key_added}_id``, ``{key_added}_label`` and ``{key_added}_method`` to ``adata.obs`` in place.

    Only distinct label/context combinations are sent to the mapper. Returns
    the per-combination mapping table.
    """
    cols = list(dict.fromkeys(c for c in (label_col, tissue_col, species_col, cell_state_col) if c))
    obs = pd.DataFrame({c: _column(adata.obs, c, c) for c in cols}, index=adata.obs.index)
    table = map_dataframe(obs.drop_duplicates().reset_index(drop=True), label_col,
                          tissue_col=tissue_col, species_col=species_col,
                          cell_state_col=cell_state_col, mapper=mapper, **mapper_kwargs)
    merged = obs.merge(table, on=cols, how="left")  # left merge keeps obs row order
    for src, dst in (("mapped_id", "id"), ("mapped_label", "label"), ("method", "method")):
        adata.obs[f"{key_added}_{dst}"] = pd.Categorical(merged[src].to_numpy())
    return table
