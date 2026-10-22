#!/usr/bin/env python3
"""Analyse routing CSVs independently of routers and model services."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

KNOWN_ROUTERS = ('CDBv1', 'CDBv2', 'CDBv3', 'LLM')


def infer_type_key(name: str) -> str | None:
    """Map paper names and legacy Types 1/3/4 to i/ii/iii; preserve legacy Type 2."""
    match = re.search(r'type[_ -](iii|ii|i)(?![a-z0-9])', name, re.I)
    if match:
        return f'type_{match.group(1).lower()}'
    match = re.search(r'type([1-4])(?![0-9])', name, re.I)
    return {'1': 'type_i', '2': 'type2', '3': 'type_ii', '4': 'type_iii'}.get(match.group(1)) if match else None


def tools(value: str) -> list[str]:
    return sorted({part.strip() for part in value.split(',') if part.strip()})


def row_metrics(row: dict) -> dict[str, float] | None:
    """Exclude unlabeled prompts; a routing failure scores zero on every metric."""
    if row['expected'] is None:
        return None
    if row['error']:
        return dict(accuracy=0.0, completeness=0.0, precision=0.0)
    expected, returned = set(row['expected']), set(row['returned'])
    precision = len(expected & returned) / len(returned) if returned else float(not expected)
    return dict(accuracy=float(expected == returned),
                completeness=float(expected <= returned), precision=precision)


def composite(a: float, c: float, p: float) -> float:
    return .25 * a + .75 * (5 * c * p / (4 * p + c) if 4 * p + c else 0.0)


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def read_run(path: Path) -> dict:
    """Read metadata-prefixed benchmark CSVs, or a plain result table."""
    metadata = {}
    with path.open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.reader(handle)
        for record in reader:
            if {'prompt', 'expected', 'returned'} <= set(record):
                header = record
                break
            if len(record) >= 2 and record[0]:
                metadata[record[0]] = record[1]
        else:
            raise ValueError(f'{path}: no result header containing prompt, expected, returned')
        records = []
        for index, record in enumerate(reader, 1):
            if not record:
                continue
            if len(record) != len(header):
                raise ValueError(f'{path}: result row {index} has an incorrect column count')
            row = dict(zip(header, record))
            flag = row.get('has_expected', 'true').strip().lower()
            if flag not in {'true', 'false', '1', '0'}:
                raise ValueError(f'{path}: invalid has_expected value in result row {index}')
            raw_latency = row.get('latency_ms', '').strip()
            latency = float(raw_latency) if raw_latency else None
            if latency is not None and (not math.isfinite(latency) or latency < 0):
                raise ValueError(f'{path}: invalid latency in result row {index}')
            records.append(dict(prompt_index=row.get('prompt_index', str(index)),
                                prompt=row['prompt'],
                                expected=tools(row['expected']) if flag in {'true', '1'} else None,
                                returned=tools(row['returned']), latency_ms=latency,
                                error=row.get('parse_error', row.get('error', '')).strip()))
    if not records:
        raise ValueError(f'{path}: empty result table')
    router = metadata.get('router', '').strip()
    if not router:
        router = next((name for parent in path.parents for name in KNOWN_ROUTERS
                       if parent.name.lower() in {name.lower(), 'router_' + name.lower()}), '')
    if not router:
        raise ValueError(f'{path}: missing router metadata and no router directory name')
    for name in KNOWN_ROUTERS:
        if router.lower() in {name.lower(), 'router_' + name.lower()}:
            router = name
    # Prefer the original input filename; timestamps and generic run names do not identify types.
    candidates = [Path(metadata.get('input_file', '')).stem, path.stem]
    candidates.extend(parent.name for parent in path.parents)
    prompt_type = next((kind for candidate in candidates
                        if (kind := infer_type_key(candidate))), 'untyped')
    return dict(source=str(path.resolve()), router=router, prompt_type=prompt_type,
                metadata=metadata, rows=records)


def load_runs(paths: list[Path]) -> list[dict]:
    """Collect every CSV once, including nested router/type/run directories."""
    files = set()
    for path in paths:
        if path.is_dir():
            files.update(p.resolve() for p in path.rglob('*.csv'))
        elif path.is_file():
            files.add(path.resolve())
        else:
            raise ValueError(f'Path not found: {path}')
    if not files:
        raise ValueError('No CSV files found')
    return [read_run(path) for path in sorted(files)]


def summarise(runs: list[dict]) -> list[dict]:
    groups = {}
    for run in runs:
        groups.setdefault((run['router'], run['prompt_type']), []).append(run)
    summaries = []
    for (router, prompt_type), group in sorted(groups.items()):
        rows = [row for run in group for row in run['rows']]
        per_run = []
        for run in group:
            metrics = [value for row in run['rows'] if (value := row_metrics(row)) is not None]
            if metrics:
                per_run.append({key: statistics.mean(m[key] for m in metrics)
                                for key in ('accuracy', 'completeness', 'precision')})
        result = dict(router=router, prompt_type=prompt_type, runs=len(group),
                      prompts=len(rows), labeled_prompts=sum(r['expected'] is not None for r in rows),
                      errors=sum(bool(r['error']) for r in rows))
        for key in ('accuracy', 'completeness', 'precision'):
            result[key] = statistics.mean(m[key] for m in per_run) if per_run else None
            result[key + '_std'] = statistics.pstdev(m[key] for m in per_run) if per_run else None
        result['L'] = composite(result['accuracy'], result['completeness'], result['precision']) if per_run else None
        latencies = [r['latency_ms'] for r in rows if r['latency_ms'] is not None]
        result.update(timed_prompts=len(latencies),
                      latency_mean_ms=statistics.mean(latencies) if latencies else None,
                      latency_median_ms=statistics.median(latencies) if latencies else None,
                      latency_p95_ms=percentile(latencies, .95),
                      latency_std_ms=statistics.pstdev(latencies) if latencies else None)
        summaries.append(result)
    return summaries


def analyse(runs: list[dict]) -> dict:
    runs = [run | {'prompt_type': infer_type_key(run['prompt_type']) or run['prompt_type']} for run in runs]
    return dict(schema_version=1, runs=runs, summary=summarise(runs))


def print_summary(summaries: list[dict]) -> None:
    print(f"{'Router':<20} {'Type':<9} {'Runs':>5} {'Rows':>6} {'Labels':>6} "
          f"{'Errors':>6} {'A':>7} {'C':>7} {'P':>7} {'L':>7} {'Mean ms':>10} {'p95 ms':>10}")
    def display(value, digits=4):
        return f'{value:.{digits}f}' if value is not None else '-'
    for row in summaries:
        print(f"{row['router']:<20} {row['prompt_type']:<9} {row['runs']:>5} {row['prompts']:>6} "
              f"{row['labeled_prompts']:>6} {row['errors']:>6} "
              + ' '.join(f'{display(row[k]):>7}' for k in ('accuracy', 'completeness', 'precision', 'L'))
              + f" {display(row['latency_mean_ms'], 3):>10} {display(row['latency_p95_ms'], 3):>10}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='CSV files or directories searched recursively')
    parser.add_argument('--output', type=Path, default=Path('analysis.json'), help='Analysis JSON (default: analysis.json)')
    args = parser.parse_args()
    try:
        report = analyse(load_runs(args.paths))
    except (ValueError, OSError, csv.Error) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print_summary(report['summary'])
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
