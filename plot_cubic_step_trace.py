#!/usr/bin/env python3
"""Plot Cubic per-step active tokens and latency from a params-metrics JSON."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp(prefix='matplotlib-', dir=tempfile.gettempdir())

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Plot Cubic step trace from cubic_params_metrics.json.',
    )
    parser.add_argument(
        'json_path',
        type=Path,
        help='Path to cubic_params_metrics.json',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Optional output PNG path. Defaults to the same directory as the JSON.',
    )
    return parser.parse_args()


def load_payload(json_path: Path) -> dict:
    with json_path.open('r', encoding='utf-8') as f:
        return json.load(f)


def load_step_trace(payload: dict, json_path: Path) -> list[dict]:
    quality_metrics = payload.get('quality_metrics') or {}
    cubic_step_trace = quality_metrics.get('cubic_step_trace') or {}
    steps = cubic_step_trace.get('steps') or []
    if not steps:
        raise ValueError(
            f'No step trace found in {json_path}. '
            'Expected quality_metrics.cubic_step_trace.steps.'
        )
    return steps


def default_output_path(json_path: Path) -> Path:
    stem = json_path.stem
    if stem.endswith('_params_metrics'):
        stem = stem[: -len('_params_metrics')]
    return json_path.parent / f'{stem}_step_trace.png'


def plot_step_trace(step_trace: list[dict], output_path: Path, title: str) -> None:
    timesteps = [int(item['step']) for item in step_trace]
    active_tokens = [int(item['active_tokens']) for item in step_trace]
    latencies = [float(item['latency_ms']) for item in step_trace]
    total_tokens = max(int(item.get('total_tokens', 0)) for item in step_trace)

    fig, ax_active = plt.subplots(figsize=(14, 7), dpi=150)
    ax_latency = ax_active.twinx()

    bar_color = '#c97b2a'
    line_color = '#1768ac'

    ax_latency.bar(
        timesteps,
        latencies,
        width=0.78,
        color=bar_color,
        alpha=0.35,
        label='Latency (ms)',
        zorder=1,
    )
    ax_active.plot(
        timesteps,
        active_tokens,
        color=line_color,
        marker='o',
        linewidth=2.2,
        markersize=4.5,
        label='Active tokens',
        zorder=3,
    )

    ax_active.set_title(title)
    ax_active.set_xlabel('Timestep')
    ax_active.set_ylabel('Active tokens', color=line_color)
    ax_latency.set_ylabel('Latency (ms)', color=bar_color)
    ax_active.tick_params(axis='y', labelcolor=line_color)
    ax_latency.tick_params(axis='y', labelcolor=bar_color)
    ax_active.set_xlim(min(timesteps) - 0.8, max(timesteps) + 0.8)
    ax_active.set_xticks(timesteps)
    ax_active.grid(axis='y', linestyle='--', alpha=0.25)

    if total_tokens > 0:
        ax_active.axhline(
            total_tokens,
            color=line_color,
            linestyle=':',
            linewidth=1.2,
            alpha=0.7,
            label=f'Total tokens = {total_tokens}',
        )

    handles_active, labels_active = ax_active.get_legend_handles_labels()
    handles_latency, labels_latency = ax_latency.get_legend_handles_labels()
    ax_active.legend(
        handles_active + handles_latency,
        labels_active + labels_latency,
        loc='upper right',
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    json_path = args.json_path.expanduser().resolve()
    if not json_path.is_file():
        raise FileNotFoundError(f'JSON not found: {json_path}')

    payload = load_payload(json_path)
    step_trace = load_step_trace(payload, json_path)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(json_path)
    )

    prompt = (payload.get('parameters') or {}).get('prompt')
    title = 'Cubic Step Trace'
    if prompt:
        title = f'Cubic Step Trace\n{prompt}'

    plot_step_trace(step_trace, output_path, title)
    print(f'Saved plot to {output_path}')


if __name__ == '__main__':
    main()
