"""Centralized visualization utilities for Cubic.

This file consolidates partition, curvature, and straightness visual exports.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from .types import Partition


class PartitionVisualizer:
    """Export partition summaries and frame-strip overlays."""

    def __init__(self, config):
        self.config = config

    def visualize_and_export(
        self,
        partition: Partition,
        curvature_map: torch.Tensor,
        block_curvatures: Dict[int, float],
        step_idx: int,
        output_dir: str,
        max_frames: int = 8,
        block_intervals: Dict[int, int] | None = None,
        file_suffix: str | None = None,
        block_value_label: str = 'tau',
    ) -> Dict[str, str]:
        if not hasattr(self, 'history_by_dir'):
            self.history_by_dir = {}
        os.makedirs(output_dir, exist_ok=True)
        if output_dir not in self.history_by_dir:
            self.history_by_dir[output_dir] = []
            
        self.history_by_dir[output_dir].append({
            'partition': partition,
            'curvature_map': curvature_map.cpu(),
            'block_curvatures': block_curvatures,
            'block_intervals': block_intervals,
            'step_idx': step_idx,
            'block_value_label': block_value_label,
        })

        suffix = ''
        if file_suffix:
            safe_suffix = str(file_suffix).replace(os.sep, '_').replace(' ', '_')
            suffix = f'_{safe_suffix}'

        summary_path = os.path.join(output_dir, f'partition_summary{suffix}.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            for rec in self.history_by_dir[output_dir]:
                summary_text = self._build_summary_text(
                    partition=rec['partition'],
                    block_curvatures=rec['block_curvatures'],
                    block_intervals=rec['block_intervals'],
                    step_idx=rec['step_idx'],
                    block_value_label=rec['block_value_label'],
                )
                f.write(summary_text + "\n========================================\n\n")

        frame_strip_path = os.path.join(output_dir, f'partition_vis{suffix}.png')
        frame_strip_frames = self._save_combined_frame_strip(
            history=self.history_by_dir[output_dir],
            save_path=frame_strip_path,
            max_frames=max_frames,
        )

        return {
            'summary_path': summary_path,
            'frame_strip_path': frame_strip_path,
            'frame_strip_frames': ','.join(str(idx) for idx in frame_strip_frames),
            'summary_text': "Written to combined summary file.",
        }

    def _build_summary_text(
        self,
        partition: Partition,
        block_curvatures: Dict[int, float],
        block_intervals: Dict[int, int] | None,
        step_idx: int,
        block_value_label: str = 'tau',
    ) -> str:
        blocks = list(partition)
        token_counts = [block.token_count for block in blocks]
        f_sizes = [block.f_size for block in blocks]
        h_sizes = [block.h_size for block in blocks]
        w_sizes = [block.w_size for block in blocks]
        curvatures = [block_curvatures.get(block.block_id, 0.0) for block in blocks]

        intervals: List[int] = []
        if block_intervals is not None:
            intervals = [int(block_intervals.get(block.block_id, 1)) for block in blocks]

        lines: List[str] = []
        lines.append(f'step={step_idx}')
        lines.append(f'partition_mode={getattr(partition, "partition_mode", "unknown")}')
        lines.append(f'is_uniform={bool(getattr(partition, "is_uniform", False))}')
        lines.append(f'selected_block_size={partition.selected_block_size}')
        lines.append(f'total_blocks={len(blocks)}')
        lines.append(f'total_tokens={partition.total_tokens}')
        lines.append('token_count_stats=' + self._format_stats(token_counts))
        lines.append('f_size_stats=' + self._format_stats(f_sizes))
        lines.append('h_size_stats=' + self._format_stats(h_sizes))
        lines.append('w_size_stats=' + self._format_stats(w_sizes))
        lines.append('curvature_stats=' + self._format_stats(curvatures, scientific=True))

        if intervals:
            configured_tau_min = int(getattr(self.config, 'tau_min', 1))
            configured_tau_max = int(
                min(
                    int(getattr(self.config, 'tau_max', 10)),
                    int(getattr(self.config, 'anchor_interval', getattr(self.config, 'tau_max', 10))),
                )
            )
            token_arr = np.asarray(token_counts, dtype=np.float64)
            tau_arr = np.asarray(intervals, dtype=np.float64)
            total_tokens = float(np.sum(token_arr))
            lines.append(f'configured_tau_min={configured_tau_min}')
            lines.append(f'configured_tau_max={configured_tau_max}')
            lines.append(f'assigned_tau_min={min(intervals)}')
            lines.append(f'assigned_tau_max={max(intervals)}')
            lines.append(f'{block_value_label}_stats=' + self._format_stats(intervals))
            if block_value_label == 'tau':
                rates = [1.0 / max(v, 1) for v in intervals]
                lines.append('update_rate_stats=' + self._format_stats(rates, scientific=True))
                weighted_tau = float(np.sum(token_arr * tau_arr) / max(total_tokens, 1.0))
                weighted_rate = float(np.sum(token_arr / np.maximum(tau_arr, 1.0)) / max(total_tokens, 1.0))
                weighted_speedup = float(1.0 / max(weighted_rate, 1e-12))
                block_rate = float(np.mean(1.0 / np.maximum(tau_arr, 1.0)))
                block_speedup = float(1.0 / max(block_rate, 1e-12))
                lines.append(f'token_weighted_tau_mean={weighted_tau:.6f}')
                lines.append(f'token_weighted_update_rate={weighted_rate:.6e}')
                lines.append(f'token_weighted_theoretical_speedup={weighted_speedup:.6f}')
                lines.append(f'block_weighted_theoretical_speedup={block_speedup:.6f}')

        lines.append('block_detail:')
        for block in blocks:
            curv = block_curvatures.get(block.block_id, 0.0)
            value_str = ''
            if block_intervals is not None:
                value_str = f' {block_value_label}={int(block_intervals.get(block.block_id, 1))}'
            lines.append(
                f'  id={block.block_id} '
                f'f=[{block.f_start},{block.f_end}) h=[{block.h_start},{block.h_end}) '
                f'w=[{block.w_start},{block.w_end}) '
                f'size=({block.f_size},{block.h_size},{block.w_size}) '
                f'tokens={block.token_count} curvature={curv:.6e}{value_str}'
            )
        return '\n'.join(lines) + '\n'

    def _save_combined_frame_strip(
        self,
        history: Sequence[Dict],
        save_path: str,
        max_frames: int,
    ) -> List[str]:
        del max_frames
        total_rows = len(history)
        if total_rows == 0:
            return []

        fig, axes = plt.subplots(
            total_rows,
            1,
            figsize=(9.2, 4.8 * total_rows),
            dpi=150,
            squeeze=False,
        )

        for row, h in enumerate(history):
            partition = h['partition']
            curvature_map = h['curvature_map'].float()
            block_intervals = h['block_intervals']
            step_idx = h['step_idx']
            block_value_label = h.get('block_value_label', 'tau')
            spatial_blocks = self._aggregate_spatial_blocks(
                partition=partition,
                block_intervals=block_intervals,
                block_value_label=block_value_label,
            )

            tau_mappable = None
            if spatial_blocks and block_intervals is not None:
                tau_values = [float(block['mean_value']) for block in spatial_blocks]
                tau_min = min(tau_values)
                tau_max = max(tau_values)
                tau_norm = Normalize(vmin=tau_min, vmax=tau_max if tau_max > tau_min else tau_min + 1.0)
                tau_mappable = ScalarMappable(norm=tau_norm, cmap='plasma')

            ax = axes[row, 0]
            mean_frame = curvature_map.mean(dim=0).cpu().numpy()
            im = ax.imshow(mean_frame, cmap='turbo', origin='upper', aspect='auto')
            mode = getattr(partition, 'partition_mode', 'unknown')
            ax.set_title(
                f'Step {step_idx:04d} mode={mode}'
            )
            ax.set_xlabel('W token')
            ax.set_ylabel('H token')

            for block in spatial_blocks:
                edgecolor = '#ffffff'
                if tau_mappable is not None and block['mean_value'] is not None:
                    edgecolor = tau_mappable.to_rgba(block['mean_value'])

                rect = Rectangle(
                    (block['w_start'], block['h_start']),
                    block['w_size'],
                    block['h_size'],
                    fill=False,
                    linewidth=2.0,
                    edgecolor=edgecolor,
                )
                ax.add_patch(rect)

                if block['label']:
                    cx = block['w_start'] + block['w_size'] / 2.0
                    cy = block['h_start'] + block['h_size'] / 2.0
                    ax.text(
                        cx,
                        cy,
                        block['label'],
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.55),
                    )

            cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
            cbar.set_label('mean curvature over frames')

            if tau_mappable is not None:
                tau_cbar = fig.colorbar(tau_mappable, ax=ax, fraction=0.018, pad=0.09)
                tau_cbar.set_label(f'mean {block_value_label}')

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return ['avg_all_frames']

    def _format_block_size_fhw(self, partition: Partition) -> str:
        selected = getattr(partition, 'selected_block_size', None)
        if selected is not None and len(selected) == 3:
            f_size, h_size, w_size = selected
            return f'({f_size},{h_size},{w_size})'

        blocks = list(partition)
        if not blocks:
            return '(unknown)'

        f_size = blocks[0].f_size
        h_size = blocks[0].h_size
        w_size = blocks[0].w_size
        return f'({f_size},{h_size},{w_size})'

    def _aggregate_spatial_blocks(
        self,
        partition: Partition,
        block_intervals: Dict[int, int] | None,
        block_value_label: str,
    ) -> List[Dict[str, int | float | str | None]]:
        grouped: Dict[tuple[int, int, int, int], Dict[str, int | float | str | None | List[int]]] = {}

        for block in partition:
            key = (block.h_start, block.h_end, block.w_start, block.w_end)
            if key not in grouped:
                grouped[key] = {
                    'h_start': block.h_start,
                    'h_size': block.h_size,
                    'w_start': block.w_start,
                    'w_size': block.w_size,
                    'values': [],
                }
            if block_intervals is not None:
                grouped[key]['values'].append(int(block_intervals.get(block.block_id, 1)))

        aggregated: List[Dict[str, int | float | str | None]] = []
        for key in sorted(grouped):
            item = grouped[key]
            values = item.pop('values')
            mean_value = None
            label = ''
            if values:
                mean_value = float(np.mean(values))
                min_value = min(values)
                max_value = max(values)
                label = (
                    f'{block_value_label}={min_value}'
                    if min_value == max_value
                    else f'{block_value_label}={min_value}-{max_value}'
                )
            item['mean_value'] = mean_value
            item['label'] = label
            aggregated.append(item)
        return aggregated

    def _format_stats(self, values: Sequence[int | float], scientific: bool = False) -> str:
        if not values:
            return 'count=0'
        arr = np.asarray(values, dtype=np.float64)
        if scientific:
            return (
                f'count={arr.size} min={arr.min():.6e} max={arr.max():.6e} '
                f'mean={arr.mean():.6e} median={np.median(arr):.6e}'
            )
        return (
            f'count={arr.size} min={arr.min():.2f} max={arr.max():.2f} '
            f'mean={arr.mean():.2f} median={np.median(arr):.2f}'
        )


class CubicVisualizer:
    """Unified visualization facade for partition/curvature/straightness."""

    def __init__(self, config):
        self.config = config
        self.partition_visualizer = PartitionVisualizer(config)

    def visualize_partition_and_frequency(
        self,
        partition: Partition,
        curvature_map: torch.Tensor,
        block_curvatures: Dict[int, float],
        block_intervals: Dict[int, int],
        step_idx: int,
        output_dir: str,
        max_frames: int = 8,
        file_suffix: str | None = None,
        block_value_label: str = 'tau',
    ) -> Dict[str, str]:
        return self.partition_visualizer.visualize_and_export(
            partition=partition,
            curvature_map=curvature_map,
            block_curvatures=block_curvatures,
            step_idx=step_idx,
            output_dir=output_dir,
            max_frames=max_frames,
            block_intervals=block_intervals,
            file_suffix=file_suffix,
            block_value_label=block_value_label,
        )

    def export_curvature_history(
        self,
        curvature_history,
        output_dir: str,
        max_frames: int = 8,
        save_raw: bool = False,
        save_strip: bool = False,
        save_meta: bool = False,
        file_suffix: str | None = None,
    ):
        pass

    def export_straightness_records(self, records, output_dir: str):
        if not records:
            return

        vis_dir = os.path.join(output_dir, 'straightness_vis')
        os.makedirs(vis_dir, exist_ok=True)
        self._clear_directory(vis_dir)

        self._save_straightness_metric_overview(records, os.path.join(vis_dir, 'straightness_overview.png'))
        self._save_straightness_heatmap_overview(records, os.path.join(vis_dir, 'straightness_heatmap_overview.png'))
        self._save_groundtruth_curvature_heatmap_overview(
            records,
            os.path.join(vis_dir, 'groundtruth_curvature_heatmap_overview.png'),
        )

    def _save_straightness_metric_overview(self, records, save_path: str):
        x_values = self._sample_flat_values(records, 'curvature_gt_map')
        y_values = self._sample_flat_values(records, 'proxy_curvature_map')
        ratios = [record.mean_ratio for record in records]
        corrs = [record.corr for record in records]

        fig = plt.figure(figsize=(13, 8), dpi=150)
        gs = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.4])

        ax_scatter = fig.add_subplot(gs[0, 0])
        if x_values.size > 0 and y_values.size > 0:
            max_points = min(3000, x_values.size)
            indices = np.linspace(0, x_values.size - 1, max_points).astype(int)
            ax_scatter.scatter(x_values[indices], y_values[indices], s=4, alpha=0.35, c='#4ea3ff')
            bound = max(float(np.max(x_values)), float(np.max(y_values)), 1e-8) * 1.05
            ax_scatter.plot([0, bound], [0, bound], linestyle='--', linewidth=1.5, color='#f0ad4e')
            ax_scatter.set_xlim(0, bound)
            ax_scatter.set_ylim(0, bound)
        ax_scatter.set_title('Proxy Curvature vs Groundtruth Curvature (Def. 3.1 approx)')
        ax_scatter.set_xlabel('groundtruth curvature')
        ax_scatter.set_ylabel('Proxy curvature')
        ax_scatter.grid(alpha=0.25)

        ax_ratio = fig.add_subplot(gs[0, 1])
        if ratios:
            xs = np.arange(len(ratios))
            ax_ratio.plot(xs, ratios, marker='o', linewidth=2.0, color='#4ea3ff')
            ax_ratio.axhline(1.0, linestyle='--', linewidth=1.5, color='#f0ad4e', label='target=1')
            ax_ratio.set_xlim(-0.2, len(ratios) - 0.8 if len(ratios) > 1 else 0.2)
        ax_ratio.set_title('Mean Ratio per Anchor')
        ax_ratio.set_xlabel('Anchor index')
        ax_ratio.set_ylabel('mean ratio')
        ax_ratio.grid(alpha=0.25)
        ax_ratio.legend(loc='best', fontsize=8)

        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis('off')
        summary = [
            f'samples={len(records)}',
            f'mean corr={np.mean(corrs):.6f}' if corrs else 'mean corr=nan',
            f'mean ratio={np.mean(ratios):.6f}' if ratios else 'mean ratio=nan',
            f'std ratio={np.std(ratios):.6f}' if ratios else 'std ratio=nan',
            f'mean proxy curvature={np.mean([r.mean_proxy_curvature for r in records]):.6e}' if records else 'mean proxy curvature=nan',
            f'mean groundtruth curvature={np.mean([r.mean_curvature_gt for r in records]):.6e}' if records else 'mean groundtruth curvature=nan',
        ]
        ax_text.text(0.01, 0.8, ' | '.join(summary), fontsize=10)

        fig.suptitle('Straightness Evaluation Overview', fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _save_groundtruth_curvature_heatmap_overview(self, records, save_path: str):
        maps = [record.curvature_gt_map.mean(dim=0).detach().cpu().numpy() for record in records]
        labels = [f'step {record.step:04d}  t={record.t:.3f}' for record in records]
        if not maps:
            return
        self._save_heatmap_grid(
            maps,
            labels,
            save_path,
            suptitle='Groundtruth Curvature Heatmap Overview (Def. 3.1 approx)',
            cmap='viridis',
            cbar_label='groundtruth curvature',
        )

    def _save_straightness_heatmap_overview(self, records, save_path: str):
        maps = [record.straightness_map.mean(dim=0).detach().cpu().numpy() for record in records]
        labels = [f'step {record.step:04d}  t={record.t:.3f}' for record in records]
        if not maps:
            return
        self._save_heatmap_grid(
            maps,
            labels,
            save_path,
            suptitle='Straightness Heatmap Overview (mean over token-time)',
            cmap='magma',
            cbar_label='straightness',
        )

    def _save_single_heatmap(self, array_2d: np.ndarray, save_path: str, title: str):
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        im = ax.imshow(array_2d, cmap='turbo', aspect='auto')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)

    def _save_heatmap_grid(
        self,
        maps: List[np.ndarray],
        labels: List[str],
        save_path: str,
        suptitle: str,
        cmap: str,
        cbar_label: str,
        force_single_row: bool = False,
    ):
        if not maps:
            raise ValueError('no maps to visualize')

        count = len(maps)
        if force_single_row:
            rows = 1
            cols = count
        else:
            cols = min(3, count)
            rows = (count + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.2 * rows), dpi=150)
        axes = np.array(axes).reshape(rows, cols)

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            if idx >= count:
                ax.axis('off')
                continue
            im = ax.imshow(maps[idx], cmap=cmap, aspect='auto')
            ax.set_title(labels[idx], fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(suptitle, fontsize=14)
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label(cbar_label)
        fig.tight_layout(rect=(0.0, 0.0, 0.98, 0.96))
        fig.savefig(save_path)
        plt.close(fig)

    def _sample_flat_values(self, records, field_name: str) -> np.ndarray:
        arrays = []
        for record in records:
            tensor = getattr(record, field_name)
            arrays.append(tensor.detach().cpu().reshape(-1).numpy())
        if not arrays:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(arrays, axis=0)

    @staticmethod
    def _clear_directory(path: str):
        for entry in os.listdir(path):
            full_path = os.path.join(path, entry)
            if os.path.isdir(full_path):
                continue
            os.remove(full_path)


GradedPartitionVisualizer = PartitionVisualizer
