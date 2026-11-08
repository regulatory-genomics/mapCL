"""Command line interface: ``mapcl map``, ``mapcl agent``, ``mapcl build-db``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _llm_args(p: argparse.ArgumentParser, optional: bool) -> None:
    from .llm import DEFAULT_MODEL, DEFAULT_REASONING_EFFORT

    p.add_argument("--model", default=os.getenv("MAPCL_MODEL", DEFAULT_MODEL),
                   help=f'LangChain "provider:model" (env MAPCL_MODEL; default {DEFAULT_MODEL})')
    p.add_argument("--reasoning-effort",
                   default=os.getenv("MAPCL_REASONING_EFFORT", DEFAULT_REASONING_EFFORT or "none"),
                   help="reasoning effort for OpenAI reasoning models, e.g. high (default %(default)s)")
    p.add_argument("--base-url", default=os.getenv("MAPCL_BASE_URL"),
                   help="API endpoint (env MAPCL_BASE_URL; else the provider's own variable)")
    if optional:
        p.add_argument("--no-llm", action="store_true",
                       help="no LLM: direct match + word-overlap search only")


def _make_llm(args):
    if getattr(args, "no_llm", False):
        return None
    from .llm import make_chat_model
    effort = None if args.reasoning_effort in ("", "none") else args.reasoning_effort
    try:
        return make_chat_model(args.model, base_url=args.base_url, reasoning_effort=effort)
    except Exception as exc:  # usually a missing API key
        raise SystemExit(f"Could not create LLM {args.model!r}: {exc}\n"
                         "Set the provider API key (e.g. DEEPSEEK_API_KEY) or use --no-llm.") from exc


def _cmd_map(args) -> int:
    from .io import map_dataframe, read_table, write_table
    from .mapper import CellOntologyMapper

    llm = _make_llm(args)
    if llm is None and (args.expand or args.no_assess):
        print("--expand/--no-assess cannot be combined with --no-llm", file=sys.stderr)
        return 2
    mapper = CellOntologyMapper(
        llm, expand=args.expand, assess=not args.no_assess, study_context=args.context,
        min_score=args.min_score, max_concurrency=args.workers,
        ols_concurrency=args.ols_workers,
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

    agent = build_agent(_make_llm(args))
    result = agent.invoke({"messages": [{"role": "user", "content": args.question}]})
    print(result["messages"][-1].content)
    return 0


def _cmd_build(args) -> int:
    from .build import build_cl_table

    path = build_cl_table(args.out_dir)
    print(f"Wrote {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mapcl", description=__doc__)
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
    m.add_argument("--no-assess", action="store_true",
                   help="skip LLM assessment (use with --expand)")
    m.add_argument("--min-score", type=int, default=6, help="lowest accepted LLM score (1-10)")
    m.add_argument("--workers", type=_positive_int, default=10,
                   help="labels processed in parallel; bounds simultaneous LLM calls (default %(default)s)")
    m.add_argument("--ols-workers", type=_positive_int, default=5,
                   help="maximum simultaneous OLS searches (default %(default)s)")
    _llm_args(m, optional=True)
    m.set_defaults(func=_cmd_map)

    a = sub.add_parser("agent", help="ask the mapping agent a question")
    a.add_argument("question")
    _llm_args(a, optional=False)
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
