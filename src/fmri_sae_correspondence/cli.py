from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

from . import __version__

RUN_STAGES = {
    "feature-banks": "feature_banks",
    "conditioning": "conditioning",
    "fitting": "fitting",
    "transfer": "transfer",
    "grouped-reliance": "grouped_reliance",
    "nulls": "nulls",
    "controls": "controls",
    "extraction": "extraction",
    "pooling": "pooling",
    "pooling-analysis": "pooling_analysis",
    "reliability": "reliability",
    "group-average": "group_average",
    "power-designs": "power_designs",
    "power": "power",
    "power-oracle": "power_oracle",
    "figures": "figures",
}


def _print_help() -> None:
    stages = ", ".join(RUN_STAGES)
    print(
        "fmri-sae-correspondence\n\n"
        "Commands:\n"
        "  config validate [--config PATH]\n"
        f"  run STAGE [stage options]    STAGE: {stages}\n"
        "  artifacts manifest [options]\n"
        "  artifacts verify [options]\n"
        "  results verify [options]\n"
        "  --version\n"
    )


def _invoke(module_name: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(f"{__package__}.{module_name}")
    function = getattr(module, "main", None)
    if not callable(function):
        raise RuntimeError(f"stage module {module_name!r} has no main(argv) entry point")
    result = function(list(argv))
    return int(result or 0)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    if arguments == ["--version"]:
        print(__version__)
        return 0

    command = arguments.pop(0)
    if command == "config" and arguments and arguments[0] == "validate":
        return _invoke("config", arguments[1:])
    if command == "run" and arguments:
        stage = arguments.pop(0)
        module_name = RUN_STAGES.get(stage)
        if module_name is None:
            print(f"unknown stage: {stage}", file=sys.stderr)
            return 2
        return _invoke(module_name, arguments)
    if command == "artifacts" and arguments:
        action = arguments.pop(0)
        modules = {"manifest": "artifact_manifest", "verify": "verification"}
        if action in modules:
            return _invoke(modules[action], arguments)
    if command == "results" and arguments and arguments.pop(0) == "verify":
        return _invoke("power_reporting", arguments)

    print(f"invalid command: {command}", file=sys.stderr)
    _print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
