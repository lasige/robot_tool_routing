#!/usr/bin/env python3
"""Generate the paper's routing figures from analysis JSON or benchmark CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from analyser import analyse, load_runs, row_metrics

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.transforms import blended_transform_factory
except ImportError as error:
    raise SystemExit('Plotting dependencies are missing. Run: python3 -m pip install -r requirements-plots.txt') from error

PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#CCB974', '#64B5CD']
COLOR_CORRECT = '#6BBD6B'
COLOR_INCORRECT = '#D96868'
HATCH_A = "\\\\"
HATCH_C = '//'
TYPE_ORDER = ['type_i', 'type_ii', 'type_iii', 'untyped']
TYPE_LABELS = {'type_i': 'Type i', 'type_ii': 'Type ii', 'type_iii': 'Type iii', 'untyped': 'All prompts'}
TypeCounts = dict[str, dict]
DPI = 300

# Figure geometry, colors, and hatching follow the repository's original paper plots.

def plot_stacked(
    all_data: dict[str, TypeCounts],
    out_path: Path,
) -> None:
    routers       = list(all_data.keys())
    n_routers     = len(routers)
    present_types = [k for k in TYPE_ORDER if any(k in d for d in all_data.values())]
    n_types       = len(present_types)

    # ── geometry ──────────────────────────────────────────────
    bar_w     = 0.10
    intra_gap = 0.02   # A ↔ C within a pair
    inter_gap = 0.09   # between router pairs
    pair_w    = 2 * bar_w + intra_gap
    total_w   = n_routers * pair_w + (n_routers - 1) * inter_gap
    pair_ctr  = np.linspace(-total_w/2 + pair_w/2,
                              total_w/2 - pair_w/2, n_routers)
    a_rel     = -(bar_w/2 + intra_gap/2)
    c_rel     =  (bar_w/2 + intra_gap/2)

    box_hw = total_w / 2 + 0.10

    matplotlib.rcParams["hatch.linewidth"] = 0.4

    fig_w = max(10, 3.2 * n_types)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    MIN_H = 1.5  # min segment height to print count inside

    def draw_bar(x: float, n_correct: float, n_wrong: float, hatch=None) -> None:
        ec = "#444444" if hatch else "white"
        ax.bar(x, n_correct, width=bar_w, color=COLOR_CORRECT,
               hatch=hatch, edgecolor=ec, linewidth=0.6)
        ax.bar(x, n_wrong, width=bar_w, bottom=n_correct,
               color=COLOR_INCORRECT, hatch=hatch, edgecolor=ec, linewidth=0.6)
        if n_correct >= MIN_H:
            ax.text(x, n_correct / 2, str(int(round(n_correct))),
                    ha="center", va="center", fontsize=9.5,
                    color="black", fontweight="bold")
        if n_wrong >= MIN_H:
            ax.text(x, n_correct + n_wrong / 2, str(int(round(n_wrong))),
                    ha="center", va="center", fontsize=9.5,
                    color="black", fontweight="bold")

    trans = ax.get_xaxis_transform()

    def label_below(x: float, y_axes: float, text: str,
                    fontsize: int = 10, color: str = "black",
                    rotation: float = 0) -> None:
        ax.text(x, y_axes, text, transform=trans,
                ha="center", va="top", fontsize=fontsize,
                color=color, clip_on=False, rotation=rotation,
                rotation_mode="anchor")

    for t_idx, type_key in enumerate(present_types):
        for r_idx, router_label in enumerate(routers):
            counts = all_data[router_label].get(type_key)
            if counts is None:
                continue

            n_tot = counts["n_total"]
            n_ca  = counts["n_correct_a"] / n_tot * 100
            n_wa  = (n_tot - counts["n_correct_a"]) / n_tot * 100
            n_cc  = counts["n_correct_c"] / n_tot * 100
            n_wc  = (n_tot - counts["n_correct_c"]) / n_tot * 100

            pc  = t_idx + pair_ctr[r_idx]
            a_x = pc + a_rel
            c_x = pc + c_rel
            draw_bar(a_x, n_ca, n_wa, hatch=HATCH_A)
            draw_bar(c_x, n_cc, n_wc, hatch=HATCH_C)
            # router name centred under the pair
            label_below(pc, -0.06, router_label, fontsize=10)

    # ── type labels (x-axis) — below router names ─────────────
    ax.tick_params(bottom=False, labelbottom=False)
    for t_idx, type_key in enumerate(present_types):
        label_below(t_idx, -0.13, TYPE_LABELS[type_key], fontsize=13)

    ax.set_ylabel("Percentage (%)", fontsize=14)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.tick_params(axis="y", labelsize=13)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.65, n_types - 0.35)

    trans_box = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(n_types):
        rect = Rectangle((i - box_hw, 0), 2 * box_hw, 1.0,
                         transform=trans_box, linewidth=1.2,
                         edgecolor="#aaaaaa", facecolor="none",
                         clip_on=True, zorder=0.5)
        ax.add_patch(rect)

    # ── legend ────────────────────────────────────────────────
    legend_handles = [
        Patch(facecolor=COLOR_CORRECT,   label="Correct"),
        Patch(facecolor=COLOR_INCORRECT, label="Incorrect"),
    ]
    legend_handles += [
        Patch(facecolor="#aaaaaa", label="Accuracy",
          hatch=HATCH_A, edgecolor="#444444"),
        Patch(facecolor="#aaaaaa", label="Completeness",
              hatch=HATCH_C, edgecolor="#444444"),
    ]
    ax.legend(handles=legend_handles, fontsize=12, framealpha=0.85)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_boxplot(
    all_data: dict[str, dict[str, list[float]]],
    out_path: Path,
):
    routers = list(all_data.keys())
    n_routers = len(routers)

    present_types = [k for k in TYPE_ORDER if any(k in d for d in all_data.values())]
    n_types = len(present_types)
    type_labels = [TYPE_LABELS[k] for k in present_types]

    box_width = 0.18
    gap = 0.05
    group_span = n_routers * box_width + (n_routers - 1) * gap
    offsets = np.linspace(
        -group_span / 2 + box_width / 2,
         group_span / 2 - box_width / 2,
        n_routers,
    )

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * n_types), 4.5))

    legend_handles = []
    for r_idx, router_label in enumerate(routers):
        color = PALETTE[r_idx % len(PALETTE)]
        for t_idx, type_key in enumerate(present_types):
            values = all_data[router_label].get(type_key, [])
            if not values:
                continue
            x_pos = t_idx + offsets[r_idx]
            bp = ax.boxplot(
                values,
                positions=[x_pos],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.75, linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markeredgecolor="none", markersize=4, alpha=0.6),
            )

        # legend handle (one per router)
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor=color, alpha=0.75, label=router_label))

    ax.set_xticks(range(n_types))
    ax.set_xticklabels(type_labels, fontsize=13)
    ax.set_ylabel("Precision", fontsize=14)

    y_lo = max(0.0, min(
        (v for d in all_data.values() for vals in d.values() for v in vals),
        default=0.0,
    ) - 0.08)
    ax.set_ylim(y_lo, 1.05)
    ax.set_xlim(-0.6, n_types - 0.4)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    box_hw = group_span / 2 + 0.10
    trans_box = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(n_types):
        rect = Rectangle((i - box_hw, 0), 2 * box_hw, 1.0,
                         transform=trans_box, linewidth=1.2,
                         edgecolor="#aaaaaa", facecolor="none",
                         clip_on=True, zorder=0.5)
        ax.add_patch(rect)

    # arrow on x-axis pointing right
    ax.annotate(
        "",
        xy=(1.015, -0.07), xycoords="axes fraction",
        xytext=(0.985, -0.07), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        annotation_clip=False,
    )

    ax.legend(handles=legend_handles, loc="lower right", fontsize=12, framealpha=0.85)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def plot_latency_by_type(
    all_data: dict[str, dict[str, list[float]]],
    out_path: Path,
    y_max: float | None = None,
    y_scale: str = 'log',
) -> None:
    if y_scale == 'log':
        omitted = sum(value <= 0 for data in all_data.values() for values in data.values() for value in values)
        all_data = {router: {kind: [value for value in values if value > 0]
                             for kind, values in data.items() if any(value > 0 for value in values)}
                    for router, data in all_data.items()}
        all_data = {router: data for router, data in all_data.items() if data}
        if omitted:
            print(f'Omitted {omitted} nonpositive latency values from the log plot; use --timing-scale linear to include them.')
        if not all_data:
            print('Skipped timing plot: no positive latency values for a logarithmic axis.')
            return
    routers       = list(all_data.keys())
    n_routers     = len(routers)
    present_types = [k for k in TYPE_ORDER if any(k in d for d in all_data.values())]
    n_types       = len(present_types)

    box_width  = 0.18
    gap        = 0.05
    group_span = n_routers * box_width + (n_routers - 1) * gap
    offsets    = np.linspace(
        -group_span / 2 + box_width / 2,
         group_span / 2 - box_width / 2,
        n_routers,
    )

    fig, ax = plt.subplots(figsize=(max(7, 2.5 * n_types), 4.5))
    ax.set_yscale(y_scale)

    legend_handles = []
    for r_idx, router_label in enumerate(routers):
        color = PALETTE[r_idx % len(PALETTE)]
        for t_idx, type_key in enumerate(present_types):
            values = all_data[router_label].get(type_key, [])
            if not values:
                continue
            x_pos = t_idx + offsets[r_idx]
            ax.boxplot(
                values,
                positions=[x_pos],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.75, linewidth=1.0),
                medianprops=dict(color="black", linewidth=1.8),
                whiskerprops=dict(color=color, linewidth=1.2),
                capprops=dict(color=color, linewidth=1.2),
                flierprops=dict(marker="o", markerfacecolor=color,
                                markeredgecolor="none", markersize=4, alpha=0.6),
            )
        legend_handles.append(Patch(facecolor=color, alpha=0.75, label=router_label))

    ax.set_xticks(range(n_types))
    ax.set_xticklabels([TYPE_LABELS[k] for k in present_types], fontsize=13)
    ax.set_ylabel("Latency (s)", fontsize=14)
    ax.tick_params(axis="y", labelsize=13)

    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(-0.6, n_types - 0.4)
    if y_max is not None:
        lower = min(ax.get_ylim()[0], y_max / 10) if y_scale == 'log' else 0
        ax.set_ylim(lower, y_max)

    box_hw    = group_span / 2 + 0.10
    trans_box = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(n_types):
        rect = Rectangle((i - box_hw, 0), 2 * box_hw, 1.0,
                         transform=trans_box, linewidth=1.2,
                         edgecolor="#aaaaaa", facecolor="none",
                         clip_on=True, zorder=0.5)
        ax.add_patch(rect)

    ax.legend(handles=legend_handles, loc="upper right", fontsize=12, framealpha=0.85)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def load_report(paths: list[Path]) -> dict:
    if len(paths) == 1 and paths[0].suffix.lower() == '.json':
        report = json.loads(paths[0].read_text(encoding='utf-8'))
        if not isinstance(report, dict) or report.get('schema_version') != 1 or not isinstance(report.get('runs'), list):
            raise ValueError('Expected an analysis.json file created by analyser.py')
        return analyse(report['runs'])
    return analyse(load_runs(paths))


def generate(report: dict, routers: list[str], output: Path, extension: str,
             timing_ymax: float | None, timing_scale: str = 'log') -> None:
    counts, precision, timing = {}, {}, {}
    for summary in report['summary']:
        router, kind = summary['router'], summary['prompt_type']
        if router not in routers or kind not in TYPE_ORDER:
            continue
        if summary['accuracy'] is not None:
            counts.setdefault(router, {})[kind] = dict(n_total=100,
                n_correct_a=100 * summary['accuracy'], n_correct_c=100 * summary['completeness'])
    for run in report['runs']:
        router, kind = run['router'], run['prompt_type']
        if router not in routers or kind not in TYPE_ORDER:
            continue
        for row in run['rows']:
            metrics = row_metrics(row)
            if metrics is not None:
                precision.setdefault(router, {}).setdefault(kind, []).append(metrics['precision'])
            if row['latency_ms'] is not None:
                timing.setdefault(router, {}).setdefault(kind, []).append(row['latency_ms'] / 1000)
    name = '_'.join(re.sub(r'[^A-Za-z0-9_.-]', '_', router) for router in routers)
    for data, function, prefix in ((counts, plot_stacked, 'accuracy_completeness'),
                                   (precision, plot_boxplot, 'precision'),
                                   (timing, plot_latency_by_type, 'timings')):
        if not data:
            print(f'Skipped {prefix}_{name}: no applicable data')
            continue
        ordered = {router: data[router] for router in routers if router in data}
        path = output / f'{prefix}_{name}.{extension}'
        if prefix == 'timings':
            function(ordered, path, y_max=timing_ymax, y_scale=timing_scale)
        else:
            function(ordered, path)


def main() -> None:
    global TYPE_ORDER, TYPE_LABELS, DPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths', nargs='+', type=Path, help='Analysis JSON, CSV files, or CSV directories')
    parser.add_argument('--output-dir', type=Path, default=Path('plots'), help='Figure directory (default: plots/)')
    parser.add_argument('--routers', nargs='+', help='Plot these routers together instead of the paper groups')
    parser.add_argument('--all-types', action='store_true', help='Also include historical Type 2 CSV data')
    parser.add_argument('--format', choices=('png', 'pdf', 'svg'), default='png', help='Figure format (default: png)')
    parser.add_argument('--dpi', type=int, default=300, help='Raster resolution (default: 300)')
    parser.add_argument('--timing-ymax', type=float, help='Optional upper latency axis limit, in seconds')
    parser.add_argument('--timing-scale', choices=('log', 'linear'), default='log',
                        help='Timing y-axis scale (default: log, matching the paper)')
    args = parser.parse_args()
    if args.dpi <= 0 or (args.timing_ymax is not None and args.timing_ymax <= 0):
        parser.error('DPI and timing axis limit must be positive')
    DPI = args.dpi
    if args.all_types:
        TYPE_ORDER = ['type_i', 'type2', 'type_ii', 'type_iii', 'untyped']
        TYPE_LABELS = TYPE_LABELS | {'type2': 'Legacy Type 2'}
    try:
        report = load_report(args.paths)
    except (ValueError, OSError, KeyError, TypeError, csv.Error) as error:
        parser.error(str(error))
    available = list(dict.fromkeys(run['router'] for run in report['runs'] if run['prompt_type'] in TYPE_ORDER))
    if not available:
        parser.error('No data remains after selecting prompt types')
    if args.routers:
        unknown = set(args.routers) - set(available)
        if unknown:
            parser.error(f'No data for routers: {", ".join(sorted(unknown))}')
        groups = [list(dict.fromkeys(args.routers))]
    else:
        groups = [list(group) for group in (('CDBv1', 'CDBv2', 'CDBv3'), ('CDBv3', 'LLM'))
                  if all(router in available for router in group)]
        covered = {router for group in groups for router in group}
        remaining = [router for router in available if router not in covered]
        if remaining:
            groups.append(remaining)
    for group in groups:
        generate(report, group, args.output_dir, args.format, args.timing_ymax, args.timing_scale)


if __name__ == '__main__':
    main()
