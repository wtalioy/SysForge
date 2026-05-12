from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass

from ..runtime import build_runtime_context, load_config
from ..workflows import build_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sysforge")
    subparsers = parser.add_subparsers(dest="workflow", required=True)
    registry = build_registry()
    for name, workflow in registry.items():
        subparsers.add_parser(name, help=workflow.description)
    return parser


def write_output(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(payload) if is_dataclass(payload) else payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config()
    context = build_runtime_context(config)
    registry = build_registry()
    workflow = registry[args.workflow]
    result = workflow.run(context)
    write_output(config.output_path, result)
    print(f"[sysforge] wrote {config.output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
