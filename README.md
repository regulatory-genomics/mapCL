# mapCL

Cell Ontology Mapper.

Map free-text cell type labels to [Cell Ontology](https://obofoundry.org/ontology/cl.html) 
(CL) terms. The tool searches the [OLS4](https://www.ebi.ac.uk/ols4) API. 
An LLM is optional: it can expand short labels and check candidates before they are accepted.

## How it works

For each distinct label + context (tissue, species, cell state):

1. **Clean** the label: drop cluster numbers ("16_", "_3", "35-"), "like",
   punctuation and plurals, and split glued words ("HypertrophicChon").
   Placeholders such as "unknown" or "NA" stop here. 
2. **Direct match** against CL labels and exact synonyms in the bundled CL
   table. Case, word order and plurals are ignored. No network call is made. 
3. **Expand** (optional, LLM): turn a label into a standard name, e.g. "KC"
   in liver becomes "Kupffer cell". Abbreviations and codes ("EC", "OGC",
   "ITL4") are expanded before the search. Full names ("Fetal Hepatic
   Endothelial") are searched and scored first, and expanded and searched
   again only if the LLM rejects every candidate.
4. **Search** OLS4 API search in CL. 
5. **Choose**
   - without an LLM: accept a term only if its label or a synonym contains
     every word of a query. 
   - with an LLM: score the candidates from 1 to 10 and accept the best one at
     `min_score` (default 6) or above. If no CL candidate passes, NCIT and BTO
     are searched and scored the same way. Otherwise the label is reported as
     unmapped.

Every row records how it was mapped (`method`), the LLM score, and a short
reason or warning (`note`).

| method | meaning |
|---|---|
| `direct` | cleaned label equals a CL label or exact synonym |
| `expanded_direct` | LLM-expanded name equals a CL label or exact synonym |
| `lexical` | no LLM: CL term containing all words of the label or a shorter form |
| `llm` | CL candidate accepted by the LLM |
| `llm_fallback` | NCIT/BTO candidate accepted by the LLM |
| `unknown` | placeholder label |
| `unmapped` | no acceptable candidate |
| `error` | OLS or LLM call failed (see `note`) |

`lexical` results are not checked by an LLM and are often broader than the
label ("Fetal Placental Fibroblast" -> fibroblast). Abbreviations are flagged
in `note`. Review these rows, or run with an LLM.

## Install

```bash
git clone https://github.com/regulatory-genomics/mapCL.git && cd mapCL
pip install -e .            # Python ≥ 3.10; use ".[anndata]" for AnnData
```

## Configure

The default LLM is DeepSeek V4.1 Flash. Set its API key:

```bash
export DEEPSEEK_API_KEY=...
export DEEPSEEK_API_BASE=...   # only if you use a DeepSeek-compatible proxy
```

For another model pass `--model`, e.g. `--model openai:gpt-6-luna` with
`OPENAI_API_KEY` set. Without any key, run with `--no-llm`.

## Usage

The input is a TSV or CSV file with one column of cell type labels.

```bash
mapcl map cells.tsv -o mapped.tsv -c cell_type --tissue-col tissue --expand
mapcl map cells.tsv -o mapped.tsv -c cell_type --no-llm
```

`--expand` lets the LLM spell out abbreviations first, which helps with
cluster labels. `mapcl map -h` lists the other options (species column, study
context, score cutoff).

In Python:

```python
from mapcl import CellOntologyMapper, make_chat_model, map_dataframe, map_adata

mapper = CellOntologyMapper(make_chat_model(), expand=True)  # CellOntologyMapper() = no LLM
out = map_dataframe(df, "cell_type", tissue_col="tissue", mapper=mapper)
map_adata(adata, "cell_type", mapper=mapper)  # adds obs cl_id, cl_label, cl_method
```

The output is the input table plus `mapped_id`, `mapped_label`,
`mapped_ontology`, `method`, `score` and `note`.

## Agent (optional)

A LangChain agent for hard labels. It wraps the fixed pipeline, using OLS search 
and CL lookup as tools. Use `mapcl map` for batch work.

```bash
mapcl agent "What CL term fits 'Mono/Mac_3' from lung? It expresses CD14 and MARCO."
```

## Reference data

`src/mapcl/data/cl_terms.tsv` holds all non-obsolete CL terms (release
2026-06-08). `mapcl build-db` updates it from OLS4.

## License and citation

MIT. Please cite the Cell Ontology (Diehl et al., *J Biomed Semantics* 2016)
and OLS (Jupp et al., *SWAT4LS* 2015).
