from __future__ import annotations

import argparse
import sys

from ..core import build_runtime_context, load_config, write_output
from ..workflows import build_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysforge")
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    registry = build_registry()
    for name, workflow in registry.items():
        subparsers.add_parser(name, help=workflow.description)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    context = build_runtime_context(config)
    registry = build_registry()
    workflow = registry.get(args.workflow)
    result = workflow.run(context)
    write_output(config.output_path, result)
    print(f"[sysforge] wrote {config.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
