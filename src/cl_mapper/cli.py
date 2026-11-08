"""Command line interface: ``clmap map``, ``clmap agent``, ``clmap build-db``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter


def _llm_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default=os.getenv("CLMAP_MODEL"),
                   help='LangChain model id, e.g. "openai:gpt-4.1-mini" (env CLMAP_MODEL)')
    p.add_argument("--base-url", default=os.getenv("CLMAP_BASE_URL"),
                   help="OpenAI-compatible endpoint (env CLMAP_BASE_URL)")


def _make_llm(args):
    if not args.model:
        return None
    from .llm import make_chat_model
    return make_chat_model(args.model, base_url=args.base_url)


def _cmd_map(args) -> int:
    from .io import map_dataframe, read_table, write_table
    from .mapper import CellOntologyMapper

    llm = _make_llm(args)
    if llm is None and (args.expand or args.no_assess):
        print("--expand/--no-assess need --model", file=sys.stderr)
        return 2
    mapper = CellOntologyMapper(
        llm, expand=args.expand, assess=not args.no_assess, study_context=args.context,
        min_score=args.min_score, max_concurrency=args.workers,
    )
    df = read_table(args.input)
    out = map_dataframe(df, args.label_col, tissue_col=args.tissue_col,
                        species_col=args.species_col, cell_state_col=args.cell_state_col,
                        mapper=mapper)
    path = write_table(out, args.output)
    counts = Counter(out["method"])
    print(f"Mapped {(out['mapped_id'].notna()).sum()}/{len(out)} rows -> {path}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in counts.most_common()))
    return 0


def _cmd_agent(args) -> int:
    from .agent import build_agent

    llm = _make_llm(args)
    if llm is None:
        print("clmap agent needs --model", file=sys.stderr)
        return 2
    agent = build_agent(llm)
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    print(result["messages"][-1].content)
    return 0


def _cmd_build(args) -> int:
    from .build import build_cl_table

    path = build_cl_table(args.out_dir)
    print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clmap", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    m = sub.add_parser("map", help="map a TSV/CSV column of cell type labels")
    m.add_argument("input", help="input .tsv or .csv")
    m.add_argument("-o", "--output", required=True, help="output .tsv or .csv")
    m.add_argument("-c", "--label-col", required=True, help="column with cell type labels")
    m.add_argument("--tissue-col")
    m.add_argument("--species-col")
    m.add_argument("--cell-state-col")
    m.add_argument("--context", help="study context for the LLM, e.g. 'human PBMC, COVID-19'")
    m.add_argument("--expand", action="store_true", help="LLM expands labels before searching")
    m.add_argument("--no-assess", action="store_true", help="skip LLM assessment; take OLS top hit")
    m.add_argument("--min-score", type=int, default=6, help="lowest accepted LLM score (1-10)")
    m.add_argument("--workers", type=int, default=4, help="labels processed in parallel")
    _llm_args(m)
    m.set_defaults(func=_cmd_map)

    a = sub.add_parser("agent", help="ask the mapping agent a question")
    a.add_argument("question")
    _llm_args(a)
    a.set_defaults(func=_cmd_agent)

    b = sub.add_parser("build-db", help="rebuild the bundled CL table from OLS4")
    b.add_argument("--out-dir", help="output folder (default: package data folder)")
    b.set_defaults(func=_cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
