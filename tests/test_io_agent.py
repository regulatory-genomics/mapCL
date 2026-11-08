import json

import pandas as pd
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from cl_mapper.agent import build_agent
from cl_mapper.cli import main
from cl_mapper.io import OUTPUT_COLUMNS, map_adata, map_dataframe, map_labels
from cl_mapper.mapper import CellOntologyMapper

from .conftest import FakeOLS, cl

ENDO = "Liver Endothelial"


@pytest.fixture
def mapper(ontology):
    ols = FakeOLS({(ENDO.lower(), "cl"): [cl("CL:1000398", "x")]})
    return CellOntologyMapper(None, ontology=ontology, ols=ols)


def test_map_dataframe_appends_columns(mapper):
    df = pd.DataFrame({"ct": ["B cells", ENDO, "unknown"], "organ": ["blood", "liver", ""]})
    out = map_dataframe(df, "ct", tissue_col="organ", mapper=mapper)
    assert list(out.columns) == ["ct", "organ", *OUTPUT_COLUMNS]
    assert out["mapped_id"].tolist()[:2] == ["CL:0000236", "CL:1000398"]
    assert pd.isna(out["mapped_id"].iloc[2])
    assert out["method"].tolist() == ["direct", "lexical", "unknown"]
    assert list(df.columns) == ["ct", "organ"]  # input untouched


def test_map_dataframe_missing_column(mapper):
    with pytest.raises(KeyError, match="tissue"):
        map_dataframe(pd.DataFrame({"ct": ["x"]}), "ct", tissue_col="organ", mapper=mapper)


def test_map_labels(mapper):
    out = map_labels(["T cells", ENDO], tissue="liver", mapper=mapper)
    assert out["mapped_id"].tolist() == ["CL:0000084", "CL:1000398"]


def test_map_adata(mapper):
    anndata = pytest.importorskip("anndata")
    obs = pd.DataFrame({"cell_type": pd.Categorical(["B cells", ENDO, "B cells", ENDO])},
                       index=[f"c{i}" for i in range(4)])
    obs["tissue"] = pd.Categorical(["blood", "liver", "blood", None])
    adata = anndata.AnnData(obs=obs)
    table = map_adata(adata, "cell_type", tissue_col="tissue", mapper=mapper)
    assert len(table) == 3
    assert adata.obs["cl_id"].tolist() == ["CL:0000236", "CL:1000398", "CL:0000236", "CL:1000398"]
    assert adata.obs["cl_method"].tolist() == ["direct", "lexical", "direct", "lexical"]
    assert list(adata.obs.index) == ["c0", "c1", "c2", "c3"]


def test_cli_map_without_llm(tmp_path, monkeypatch, ontology):
    monkeypatch.setattr("cl_mapper.mapper.load_bundled", lambda: ontology)
    monkeypatch.setattr("cl_mapper.mapper.OLSClient", lambda: FakeOLS())
    src = tmp_path / "in.tsv"
    pd.DataFrame({"label": ["B cells", "Treg"]}).to_csv(src, sep="\t", index=False)
    assert main(["map", str(src), "-o", str(tmp_path / "out.tsv"), "-c", "label", "--no-llm"]) == 0
    out = pd.read_csv(tmp_path / "out.tsv", sep="\t")
    assert out["mapped_id"].tolist() == ["CL:0000236", "CL:0000815"]


def test_cli_expand_needs_model(tmp_path, monkeypatch):
    monkeypatch.delenv("CLMAP_MODEL", raising=False)
    assert main(["map", "x.tsv", "-o", "y.tsv", "-c", "l", "--expand", "--no-llm"]) == 2


class ToolFakeChat(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_agent_calls_mapping_tool(ontology):
    llm = ToolFakeChat(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "map_cell_label", "args": {"label": "B cells"},
                                           "id": "call_1"}]),
        AIMessage(content="B cells -> CL:0000236 B cell"),
    ]))
    agent = build_agent(llm, ontology=ontology, ols=FakeOLS())
    result = agent.invoke({"messages": [{"role": "user", "content": "Map 'B cells'"}]})
    tool_msg = next(m for m in result["messages"] if m.type == "tool")
    assert json.loads(tool_msg.content)["mapped_id"] == "CL:0000236"
    assert result["messages"][-1].content.endswith("B cell")
