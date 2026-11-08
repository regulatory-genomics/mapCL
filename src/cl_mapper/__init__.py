"""Map free-text cell type labels to Cell Ontology terms."""

from .io import map_adata, map_dataframe, map_labels
from .llm import make_chat_model
from .mapper import METHODS, CellOntologyMapper, MappingResult, Query
from .ols import OLSClient
from .ontology import CellOntology, Term, load_bundled

__all__ = [
    "CellOntology", "CellOntologyMapper", "METHODS", "MappingResult", "OLSClient", "Query",
    "Term", "load_bundled", "make_chat_model", "map_adata", "map_dataframe", "map_labels",
]
__version__ = "0.1.0"
