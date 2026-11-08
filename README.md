# CellOntologyMapper

Map free-text cell type labels ("16_T cells", "KC", "Liver endothelial") to
[Cell Ontology](https://obofoundry.org/ontology/cl.html) (CL) terms. The tool
searches the [OLS4](https://www.ebi.ac.uk/ols4) API. An LLM is optional: it can
expand short labels and check candidates before they are accepted.

## How it works

For each distinct label + context (tissue, species, cell state):

1. **Clean** the label: drop cluster numbers, "like", punctuation and plurals.
   Placeholders such as "unknown" or "NA" stop here.
2. **Direct match** against CL labels and exact synonyms in the bundled CL
   table. Case, word order and plurals are ignored. No network call is made.
3. **Expand** (optional, LLM): turn abbreviations into a standard name,
   e.g. "KC" in liver becomes "Kupffer cell".
4. **Search** OLS4 in CL.
5. **Choose**
   - without an LLM: take the top CL hit;
   - with an LLM: score the candidates from 1 to 10 and accept the best one at
     `min_score` (default 6) or above. If no CL candidate passes, NCIT and BTO
     are searched and scored the same way. Otherwise the label is reported as
     unmapped.

Every row records how it was mapped (`method`), the LLM score, and the LLM's
one-line reason (`note`).

| method | meaning |
|---|---|
| `direct` | cleaned label equals a CL label or exact synonym |
| `expanded_direct` | LLM-expanded name equals a CL label or exact synonym |
| `ols_top_hit` | first CL search hit, not checked by an LLM |
| `llm` | CL candidate accepted by the LLM |
| `llm_fallback` | NCIT/BTO candidate accepted by the LLM |
| `unknown` | placeholder label |
| `unmapped` | no acceptable candidate |
| `error` | OLS or LLM call failed (see `note`) |

`ols_top_hit` results are unchecked. Review them, or run with an LLM.

## Install

Python 3.10 or newer.

```bash
git clone <repo-url> cell-ontology-mapper
cd cell-ontology-mapper
pip install -e .            # add ".[anndata]" for AnnData support
```

LLM steps use LangChain's `init_chat_model`. OpenAI support is included. For
other providers install their package (e.g. `pip install langchain-anthropic`)
and pass `"anthropic:<model>"`. Keys are read from the provider's usual
environment variable (`OPENAI_API_KEY`, ...). See `.env.example`.

## Usage

### Command line

```bash
# No LLM: direct match + OLS top hit
clmap map examples/labels.tsv -o mapped.tsv -c cell_type --tissue-col tissue

# With LLM assessment (and optional expansion)
export OPENAI_API_KEY=...
clmap map examples/labels.tsv -o mapped.tsv -c cell_type \
    --tissue-col tissue --species-col species \
    --model openai:gpt-4.1-mini --expand --context "human scRNA-seq atlas"
```

The output is the input table plus `mapped_id`, `mapped_label`,
`mapped_ontology`, `method`, `score`, `expanded_label`, `cleaned_label` and
`note`. Run `clmap map -h` for all options.

### Python

```python
import pandas as pd
from cl_mapper import CellOntologyMapper, map_dataframe, map_labels, map_adata

# Quick list, no LLM
map_labels(["T cells", "Kupffer cells", "Liver endothelial"], tissue="liver")

# With an LLM
mapper = CellOntologyMapper("openai:gpt-4.1-mini", expand=True,
                            study_context="human liver atlas")
df = pd.read_csv("examples/labels.tsv", sep="\t")
out = map_dataframe(df, "cell_type", tissue_col="tissue",
                    species_col="species", mapper=mapper)

# AnnData: adds obs["cl_id"], obs["cl_label"], obs["cl_method"]
table = map_adata(adata, "cell_type", tissue_col="tissue", mapper=mapper)
```

`CellOntologyMapper` takes a LangChain chat model object or a
`"provider:model"` string. For a local OpenAI-compatible server:

```python
from cl_mapper import make_chat_model
llm = make_chat_model("openai:qwen2.5-72b", base_url="http://localhost:8000/v1", api_key="none")
```

### Agent (optional)

For interactive work on hard labels there is a small LangChain agent
(`create_agent`). Its tools are the pipeline itself, OLS search, and CL term
lookup. The agent may only report IDs that a tool returned. Use the pipeline
above for batch mapping, because its results are reproducible.

```bash
clmap agent "What CL term fits 'Mono/Mac_3' from lung? It expresses CD14 and MARCO." \
    --model openai:gpt-4.1-mini
```

```python
from cl_mapper.agent import build_agent
agent = build_agent("openai:gpt-4.1-mini")
agent.invoke({"messages": [{"role": "user", "content": "Map 'cDC1' from mouse spleen"}]})
```

## Reference data

`src/cl_mapper/data/cl_terms.tsv` holds every non-obsolete CL class: ID,
label, IAO definition, exact synonyms and other synonyms. It is used for
direct matching and to give the LLM clean definitions.
`cl_terms.meta.json` records the CL release (currently 2026-06-08). To update
the table to the CL release that OLS currently serves:

```bash
clmap build-db
```

## Development

```bash
pip install -e ".[dev]"
pytest                          # offline: fake OLS and fake LLM
CLMAP_LIVE=1 pytest             # also run the checks against the live OLS4 API
```

```
src/cl_mapper/
  text.py      label cleaning and match keys
  ontology.py  bundled CL table and direct-match index
  ols.py       OLS4 client (search, retries, cache, table download)
  llm.py       prompts, structured outputs, chat-model factory
  mapper.py    the mapping pipeline
  io.py        DataFrame / file / AnnData helpers
  agent.py     optional create_agent helper
  build.py     rebuild the CL table
  cli.py       clmap command
```

## License and citation

MIT. Please cite the Cell Ontology (Diehl et al., *J Biomed Semantics* 2016)
and OLS (Jupp et al., *SWAT4LS* 2015).
