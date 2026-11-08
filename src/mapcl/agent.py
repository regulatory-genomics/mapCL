"""Optional conversational helper built with LangChain ``create_agent``.

The agent wraps the fixed pipeline as tools. Use it for interactive work on
hard labels ("what could 'Mono/Mac_3' in lung be?"). Batch mapping should go
through :class:`~mapcl.mapper.CellOntologyMapper`, which is deterministic.
"""

from __future__ import annotations

import json

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

from . import llm as llm_mod
from .mapper import CellOntologyMapper, Query
from .ols import OLSClient
from .ontology import CellOntology, load_bundled

SYSTEM_PROMPT = """You help researchers map cell type labels to Cell Ontology (CL) terms.
Tools:
- map_cell_label runs the standard mapping pipeline; try it first.
- search_ontology searches OLS (CL by default; NCIT or BTO when CL has no fit).
- get_cl_term returns the label, definition and synonyms of a CL ID.
Rules: only report IDs that a tool returned. Give the ID, the label and a one-line
reason. If nothing fits, say so and suggest the closest broader CL term."""


def build_agent(
    llm: BaseChatModel | str = llm_mod.DEFAULT_MODEL,
    *,
    mapper: CellOntologyMapper | None = None,
    ontology: CellOntology | None = None,
    ols: OLSClient | None = None,
    **agent_kwargs,
):
    """Return a LangGraph agent. Invoke with ``{"messages": [{"role": "user", "content": ...}]}``."""
    if isinstance(llm, str):
        llm = llm_mod.make_chat_model(llm)
    ontology = ontology or load_bundled()
    ols = ols or OLSClient()
    mapper = mapper or CellOntologyMapper(llm, ontology=ontology, ols=ols)

    @tool
    def map_cell_label(label: str, tissue: str = "", species: str = "", cell_state: str = "") -> str:
        """Map one cell type label to the Cell Ontology with the standard pipeline."""
        res = mapper.map([Query(label, tissue, species, cell_state)])[0]
        return json.dumps(res.to_dict())

    @tool
    def search_ontology(query: str, ontology: str = "cl", rows: int = 8) -> str:
        """Search OLS4 for classes. ontology is a comma-separated list of 'cl', 'ncit', 'bto'."""
        names = tuple(o.strip().lower() for o in ontology.split(",") if o.strip()) or ("cl",)
        hits = ols.search(query, names, rows=max(1, min(rows, 20)))
        return json.dumps([h.to_dict() for h in hits])

    @tool
    def get_cl_term(cl_id: str) -> str:
        """Look up a CL term by ID, e.g. 'CL:0000236'."""
        term = ontology.get(cl_id)
        return json.dumps(term.to_dict() if term else {"error": f"{cl_id} not found"})

    return create_agent(llm, tools=[map_cell_label, search_ontology, get_cl_term],
                        system_prompt=SYSTEM_PROMPT, **agent_kwargs)
