#!/usr/bin/env python3
"""Run routers over prompt files and record selections, labels, timings and errors.

Examples (from the repository root):
  python3 bench.py CDBv3 LLM --prompt-dir prompts/
  python3 bench.py CDBv1 CDBv2 --prompts prompts/type_i.json --runs 5
  python3 bench.py CDBv3 --explain "monitor the battery"

No plots or quality metrics are calculated. CSVs retain expected labels and
returned tool names for comparison by a separate caller.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import importlib.util
import inspect
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ROUTING_DIR = PROJECT_DIR / "Routing"
DEFAULT_CONFIG = PROJECT_DIR / "config.json"
DEFAULT_CONTEXT = ROUTING_DIR / "context.json"
RESULTS_DIR = PROJECT_DIR / "output"

for path in (PROJECT_DIR, ROUTING_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TYPE_KEYS = ["type_i", "type_ii", "type_iii"]

CSV_METADATA_FIELDS = [
    "experiment_timestamp",
    "router",
    "input_file",
    "config_file",
    "context_file",
    "output_file",
    "threshold",
    "chromadb_n_results",
    "model",
    "total_duration_s",
]

CSV_ROW_COLUMNS = [
    "prompt_index",
    "prompt",
    "expected",
    "has_expected",
    "returned",
    "distances",
    "latency_ms",
    "raw_response",
    "parse_error",
]


# ─────────────────────────────────────────────────────────────
# Router discovery and construction
# ─────────────────────────────────────────────────────────────

def discover_routers() -> dict[str, Path]:
    """Map router name (file stem without 'router_') → file path."""
    return {
        path.stem.removeprefix("router_"): path
        for path in sorted(ROUTING_DIR.glob("router_*.py"))
    }


def resolve_router_name(name: str, available: dict[str, Path]) -> str:
    """Match a router name case-insensitively, with or without the 'router_' prefix."""
    if name in available:
        return name
    lowered = {key.lower(): key for key in available}
    key = name.lower().removeprefix("router_")
    if key in lowered:
        return lowered[key]
    raise SystemExit(f"Unknown router '{name}'. Available: {', '.join(available)}")


def load_module(path: Path):
    """Import a router file by path ('+' is not valid in a module name)."""
    module_name = path.stem.replace("+", "_plus_")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load routing script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_value(text: str):
    """Turn an override value from the command line into bool, int, float or str."""
    lowered = text.strip().lower()
    if lowered in {"true", "on", "yes"}:
        return True
    if lowered in {"false", "off", "no"}:
        return False
    for caster in (int, float):
        try:
            return caster(text)
        except ValueError:
            pass
    return text


@dataclass
class RouterSpec:
    """A router to benchmark: its name plus optional overrides (NAME:key=value,...)."""
    name: str
    overrides: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Display name used in terminal output and CSV metadata."""
        if not self.overrides:
            return self.name
        parts = ",".join(f"{k}={str(v).lower()}" for k, v in self.overrides.items())
        return f"{self.name}:{parts}"

    @property
    def folder(self) -> str:
        """Filesystem-safe version of the label (':' and '=' are not valid on Windows)."""
        return re.sub(r"[^A-Za-z0-9_.+-]+", "_", self.label.replace(":", "__").replace("=", "-"))


def parse_spec(raw: str, available: dict[str, Path]) -> RouterSpec:
    """Parse 'NAME' or 'NAME:key=value,key=value' into a RouterSpec."""
    name, _, rest = raw.partition(":")
    overrides = {}
    for item in filter(None, (part.strip() for part in rest.split(","))):
        if "=" not in item:
            raise SystemExit(f"Bad override '{item}' in '{raw}' (expected key=value)")
        key, value = item.split("=", 1)
        overrides[key.strip()] = parse_value(value)
    return RouterSpec(resolve_router_name(name.strip(), available), overrides)


@dataclass
class Runner:
    """A built router ready to evaluate prompts with the same call for every router type."""
    spec: RouterSpec
    evaluate: callable
    is_llm: bool
    model: str
    threshold: object
    router: object = None


def build_runner(spec: RouterSpec, available: dict[str, Path], config: dict) -> Runner:
    """Build the router once and wrap it in an evaluate(prompt) function.

    Pure-LLM routers (router_LLM) have their own setup; every other router is
    built through its build_router(), passing only the arguments it accepts.
    """
    module = load_module(available[spec.name])

    # pure LLM router: no Router object in the evaluation path
    if hasattr(module, "build_routing_context"):
        if spec.overrides:
            raise SystemExit(f"{spec.name} does not accept overrides")
        context = json.loads(DEFAULT_CONTEXT.read_text(encoding="utf-8"))
        routing_context = module.build_routing_context(config, context, config["llm"]["routing_model"])
        llm = module.build_llm(config, routing_context["model"])

        def evaluate(prompt: str) -> dict:
            routed = module.route_prompt(prompt, llm, routing_context)
            return {
                "returned": routed["selected_tools"],
                "distances": [],
                "raw_response": routed.get("raw_response", ""),
                "parse_error": routed.get("parse_error", ""),
            }

        return Runner(spec, evaluate, True, routing_context["model"], "")

    accepted = inspect.signature(module.build_router).parameters
    kwargs = {}
    if "context_path" in accepted:
        kwargs["context_path"] = DEFAULT_CONTEXT
    if spec.overrides:
        if "overrides" not in accepted:
            raise SystemExit(f"{spec.name} does not accept overrides")
        kwargs["overrides"] = spec.overrides

    router = module.build_router(config, **kwargs)
    threshold = getattr(module, "THRESHOLD_LABEL", config["routing"]["distance_threshold"])

    def evaluate(prompt: str) -> dict:
        routed = module.route_prompt(prompt, router, config["routing"]["distance_threshold"])
        return {
            "returned": routed["selected_tools"],
            "distances": routed.get("distances", []),
            "raw_response": routed.get("raw_response", ""),
            "parse_error": routed.get("parse_error", ""),
        }

    is_llm = getattr(router, "uses_llm", hasattr(router, "llm"))
    return Runner(spec, evaluate, is_llm, getattr(router, "model", ""), threshold, router)


# ─────────────────────────────────────────────────────────────
# Prompt input
# ─────────────────────────────────────────────────────────────

def infer_type_key(name: str) -> str | None:
    """Recognize paper type names and historical numeric filenames."""
    match = re.search(r"type[_ -](iii|ii|i)(?![a-z0-9])", name, re.I)
    if match:
        return f"type_{match.group(1).lower()}"
    match = re.search(r"type([134])(?![0-9])", name, re.I)
    return {"1": "type_i", "3": "type_ii", "4": "type_iii"}.get(match.group(1)) if match else None


def prompt_sets(types: list[str] | None, prompt_files: list[str] | None,
                prompt_dir: str | None = None) -> list[tuple[str, Path]]:
    """Return [(type_key, prompt_file)] from an explicit directory or file list."""
    if prompt_dir:
        directory = Path(prompt_dir)
        if not directory.is_dir():
            raise SystemExit(f"Prompt directory not found: {prompt_dir}")
        sets = []
        for kind in [f"type_{value}" for value in types] if types else TYPE_KEYS:
            matches = sorted(path for path in directory.glob("*.json")
                             if infer_type_key(path.stem) == kind)
            if matches:
                sets.append((kind, matches[0].resolve()))
        if not sets:
            raise SystemExit(f"No type_i.json, type_ii.json, or type_iii.json files in {prompt_dir}")
        return sets

    if prompt_files:
        sets = []
        for raw in prompt_files:
            path = Path(raw)
            if not path.exists():
                raise SystemExit(f"Prompt file not found: {raw}")
            sets.append((infer_type_key(path.stem) or path.stem, path.resolve()))
        return sets

    raise SystemExit("Supply --prompts FILE ... or --prompt-dir DIR (for example, --prompt-dir prompts/).")


def load_prompts(path: Path, limit: int | None) -> list[dict]:
    """Read a prompt file (strings or {"prompt", "expected_tools"} objects)."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for entry in entries:
        if isinstance(entry, str):
            prompts.append({"prompt": entry, "expected": [], "has_expected": False})
            continue
        expected = entry.get("expected_tools", [])
        if isinstance(expected, str):
            expected = [expected]
        prompts.append({"prompt": entry["prompt"], "expected": expected or [],
                        "has_expected": "expected_tools" in entry and expected is not None})
    return prompts[:limit] if limit else prompts


# ─────────────────────────────────────────────────────────────
# Running and saving
# ─────────────────────────────────────────────────────────────

def run_once(runner: Runner, prompts: list[dict], verbose: bool) -> list[dict]:
    """Route every prompt once. A router exception is recorded as a parse error, not a crash."""
    rows = []
    for index, entry in enumerate(prompts, start=1):
        t0 = time.perf_counter()
        try:
            result = runner.evaluate(entry["prompt"])
        except Exception as error:
            result = {"returned": [], "distances": [], "raw_response": "",
                      "parse_error": f"exception: {error}"}
        latency_ms = round((time.perf_counter() - t0) * 1000, 3)

        rows.append({
            "prompt_index": index,
            "prompt": entry["prompt"],
            "expected": ",".join(entry["expected"]),
            "has_expected": entry["has_expected"],
            "returned": ",".join(result["returned"]),
            "distances": ",".join(str(d) for d in result["distances"]),
            "latency_ms": latency_ms,
            "raw_response": result.get("raw_response", ""),
            "parse_error": result.get("parse_error", ""),
        })
        if verbose:
            print(f"    [{index}] {entry['prompt']} ({latency_ms} ms)")
            print(f"        expected: {', '.join(entry['expected'])}")
            print(f"        returned: {', '.join(result['returned']) or '(none)'}")
            if result.get("parse_error"):
                print(f"        error: {result['parse_error']}")
    return rows


def write_csv(rows: list[dict], path: Path, metadata: dict):
    """Write metadata rows, a blank line, then the result table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for key in CSV_METADATA_FIELDS:
            writer.writerow([key, metadata.get(key, "")])
        writer.writerow([])
        dict_writer = csv.DictWriter(f, fieldnames=CSV_ROW_COLUMNS)
        dict_writer.writeheader()
        for row in rows:
            dict_writer.writerow({k: row[k] for k in CSV_ROW_COLUMNS})


def rel(path: Path) -> str:
    """Path relative to the repository root (absolute if outside it)."""
    try:
        return path.resolve().relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_explain(runner: Runner, prompt: str):
    """Route one prompt and print returned tools, distances and errors."""
    print(f"\n### {runner.spec.label}")
    result = runner.evaluate(prompt)
    pairs = zip(result["returned"], result["distances"] or [""] * len(result["returned"]))
    print("  returned: " + (", ".join(f"{n} {d}".strip() for n, d in pairs) or "(none)"))
    if result.get("parse_error"):
        print(f"  parse_error: {result['parse_error']}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    """Build each router once, then record each prompt selection and its timing."""
    parser = argparse.ArgumentParser(
        description="Record router selections, expected labels, timings and errors as CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("routers", nargs="+", metavar="ROUTER",
                        help="Router names: CDBv1, CDBv2, CDBv3, LLM")
    parser.add_argument("--types", nargs="+", choices=["i", "ii", "iii"],
                        help="Prompt types to run (default: all)")
    parser.add_argument("--prompts", nargs="+", metavar="FILE",
                        help="Custom prompt JSON file(s) instead of the typed sets")
    parser.add_argument("--prompt-dir", metavar="DIR",
                        help="Directory with type_i.json, type_ii.json, and type_iii.json")
    parser.add_argument("--limit", type=int, metavar="N", help="Only the first N prompts of each set")
    parser.add_argument("--runs", type=int, default=1, metavar="N",
                        help="Runs per prompt set for every selected router (default: 1)")
    parser.add_argument("--explain", metavar="PROMPT", help="Show the routing decision for one prompt and exit")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.json")
    parser.add_argument("--out", metavar="DIR", help=f"Output directory (default: {rel(RESULTS_DIR)}/<timestamp>)")
    parser.add_argument("--no-save", action="store_true", help="Print per-prompt results without writing CSVs")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-prompt output and router logs")
    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    # silence router logs before any router module binds logger.log
    if not args.verbose:
        import logger
        logger.log = lambda *a, **k: None

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    available = discover_routers()
    specs = [parse_spec(raw, available) for raw in args.routers]

    # ── explain mode ────────────────────────────────────────
    if args.explain:
        print(f"PROMPT: {args.explain}")
        for spec in specs:
            print_explain(build_runner(spec, available, config), args.explain)
        return

    sets = prompt_sets(args.types, args.prompts, args.prompt_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else RESULTS_DIR / timestamp

    print(f"Routers : {', '.join(s.label for s in specs)}")
    print(f"Sets    : {', '.join(f'{k} ({rel(p)})' for k, p in sets)}")
    if not args.no_save:
        print(f"Output  : {rel(out_dir)}")

    for spec in specs:
        print(f"\n> {spec.label}")
        try:
            runner = build_runner(spec, available, config)
        except Exception as error:
            print(f"  FAILED to build router: {error}")
            continue

        n_runs = args.runs
        for key, path in sets:
            prompts = load_prompts(path, args.limit)
            for run_index in range(1, n_runs + 1):
                t0 = time.perf_counter()
                rows = run_once(runner, prompts, args.verbose or args.no_save)
                duration_s = round(time.perf_counter() - t0, 2)

                if not args.no_save:
                    csv_path = out_dir / spec.folder / key / f"run_{run_index:02d}.csv"
                    write_csv(rows, csv_path, {
                        "experiment_timestamp": timestamp,
                        "router": spec.label,
                        "input_file": rel(path),
                        "config_file": rel(Path(args.config)),
                        "context_file": rel(DEFAULT_CONTEXT),
                        "output_file": rel(csv_path),
                        "threshold": runner.threshold,
                        "chromadb_n_results": config["routing"].get("chromadb_n_results") or "all",
                        "model": runner.model,
                        "total_duration_s": duration_s,
                    })

                run_tag = f" run {run_index}/{n_runs}" if n_runs > 1 else ""
                print(f"  {key}{run_tag}: recorded {len(rows)} prompts (total={duration_s}s)")

    if not args.no_save:
        print(f"Results directory: {rel(out_dir)}")


if __name__ == "__main__":
    main()
