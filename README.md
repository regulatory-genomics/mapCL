# CellOntologyMapper

Map free-text cell type labels ("16_T cells", "KC", "Liver endothelial") to
[Cell Ontology](https://obofoundry.org/ontology/cl.html) (CL) terms. The tool
searches the [OLS4](https://www.ebi.ac.uk/ols4) API. An LLM is optional: it can
expand short labels and check candidates before they are accepted.

## How it works

For each distinct label + context (tissue, species, cell state):

1. **Clean** the label: drop cluster numbers ("16_", "_3", "35-"), "like",
   punctuation and plurals, and split glued words ("HypertrophicChon").
   Placeholders such as "unknown" or "NA" stop here. Labels that clean to the
   same text ("Fetal Astrocyte 1", "Fetal Astrocyte 2") are mapped once.
2. **Direct match** against CL labels and exact synonyms in the bundled CL
   table, also with "cell" added ("Plasma" -> plasma cell). Case, word order
   and plurals are ignored. No network call is made. With an LLM, short
   abbreviations ("IC", "PC") only become candidates, because the CL synonym
   is often a different cell type.
3. **Expand** (optional, LLM): turn abbreviations into a standard name,
   e.g. "KC" in liver becomes "Kupffer cell".
4. **Search** OLS4 in CL. OLS needs every query word to match, so shorter
   forms are searched too: without "fetal"/"cycling", without the first word,
   or with a bracketed qualifier moved in front ("Fibroblast (Splenic)" ->
   "Splenic Fibroblast"). Terms restricted to another species are dropped.
5. **Choose**
   - without an LLM: accept a term only if its label or a synonym contains
     every word of a query. The term with the fewest extra words wins.
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

Python 3.10 or newer.

```bash
git clone <repo-url> cell-ontology-mapper
cd cell-ontology-mapper
pip install -e .            # add ".[anndata]" for AnnData support
```

LLM steps use LangChain's `init_chat_model`. The default model is
**DeepSeek V4.1 Flash** (`deepseek:deepseek-flash`; DeepSeek serves V4.1
Flash under the API name `deepseek-flash`). It reads `DEEPSEEK_API_KEY`, and
`DEEPSEEK_API_BASE` for a DeepSeek-compatible proxy. DeepSeek and OpenAI
support are included. For other providers, install their package (e.g.
`pip install langchain-anthropic`) and pass `"anthropic:<model>"`. See
`.env.example`.

## Usage

### Command line

```bash
# Default: DeepSeek V4.1 Flash assesses the candidates
export DEEPSEEK_API_KEY=...
clmap map examples/labels.tsv -o mapped.tsv -c cell_type \
    --tissue-col tissue --species-col species \
    --expand --context "human scRNA-seq atlas"

# Another model
clmap map examples/labels.tsv -o mapped.tsv -c cell_type \
    --model openai:gpt-6-luna --reasoning-effort high

# No LLM: direct match + word-overlap search
clmap map examples/labels.tsv -o mapped.tsv -c cell_type --no-llm
```

The output is the input table plus `mapped_id`, `mapped_label`,
`mapped_ontology`, `method`, `score`, `expanded_label`, `cleaned_label` and
`note`. Run `clmap map -h` for all options.

### Python

```python
import pandas as pd
from cl_mapper import CellOntologyMapper, map_dataframe, map_labels, map_adata

# Quick list, no LLM (the Python API uses an LLM only when you pass one)
map_labels(["T cells", "Kupffer cells", "Liver endothelial"], tissue="liver")

# With the default LLM (DeepSeek V4.1 Flash)
from cl_mapper import make_chat_model
mapper = CellOntologyMapper(make_chat_model(), expand=True,
                            study_context="human liver atlas")
df = pd.read_csv("examples/labels.tsv", sep="\t")
out = map_dataframe(df, "cell_type", tissue_col="tissue",
                    species_col="species", mapper=mapper)

# AnnData: adds obs["cl_id"], obs["cl_label"], obs["cl_method"]
table = map_adata(adata, "cell_type", tissue_col="tissue", mapper=mapper)
```

`CellOntologyMapper` takes a LangChain chat model object or a
`"provider:model"` string. `make_chat_model()` returns the default model.
Other models:

```python
make_chat_model("openai:gpt-6-luna", reasoning_effort="high")
make_chat_model("anthropic:claude-sonnet-5")
make_chat_model("openai:qwen2.5-72b", base_url="http://localhost:8000/v1",
                api_key="none")   # local OpenAI-compatible server
```

`reasoning_effort` only applies to OpenAI reasoning models.

### Agent (optional)

For interactive work on hard labels there is a small LangChain agent
(`create_agent`). Its tools are the pipeline itself, OLS search, and CL term
lookup. The agent may only report IDs that a tool returned. Use the pipeline
above for batch mapping, because its results are reproducible.

```bash
clmap agent "What CL term fits 'Mono/Mac_3' from lung? It expresses CD14 and MARCO."
```

```python
from cl_mapper.agent import build_agent
agent = build_agent()  # default model
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
