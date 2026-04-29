import os

f_path = '/home/chenyy/WORK/cyy/cubic/cubic_core/visualizer.py'

with open(f_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace visualizer.py lines to only use one summary file and one png file directly in output_dir

import re

# find exactly where visualize_and_export starts
start_str = "    def visualize_and_export("
if start_str not in text:
    print("Cannot find start_str")

new_methods = """    def visualize_and_export(
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
                f.write(self._build_summary_text(
                    partition=rec['partition'],
                    block_curvatures=rec['block_curvatures'],
                    block_intervals=rec['block_intervals'],
                    step_idx=rec['step_idx'],
                    block_value_label=rec['block_value_label'],
                ))
                f.write("\\n============================================================\\n\\n")

        frame_strip_path = os.path.join(output_dir, f'partition_vis{suffix}.png')
        frame_strip_frames = self._save_combined_frame_strip(
            history=self.history_by_dir[output_dir],
            save_path=frame_strip_path,
            max_frames=max_frames,
        )

        return {
            'summary_path': summary_path,
            'frame_strip_path': frame_strip_path,
            'frame_strip_frames': ','.join(str(idx) for idx in frame_strip_frames) if frame_strip_frames else "",
            'summary_text': "Written to combined summary file.",
        }

    def _save_combined_frame_strip(
        self,
        history: Sequence[Dict],
        save_path: str,
        max_frames: int,
    ) -> List[int]:
        total_rows = 0
        frame_info_list = []
        last_frame_indices = []
        for h in history:
            curvature_map = h['curvature_map']
            ft, _, _ = curvature_map.shape
            frame_count = min(max_frames, ft)
            frame_indices = torch.linspace(0, ft - 1, frame_count).round().long().tolist()
            if not last_frame_indices:
                last_frame_indices = frame_indices
            total_rows += len(frame_indices)
            frame_info_list.append((h, frame_indices))
            
        if total_rows == 0:
            return last_frame_indices

        fig, axes = plt.subplots(
            total_rows,
            1,
            figsize=(9.2, 3.8 * total_rows),
            dpi=150,
            squeeze=False,
        )

        current_row = 0
        for h, frame_indices in frame_info_list:
            partition = h['partition']
            curvature_map = h['curvature_map']
            block_intervals = h['block_intervals']
            step_idx = h['step_idx']
            block_value_label = h.get('block_value_label', 'tau')
            
            blocks = list(partition)

            tau_norm = None
            tau_mappable = None
            if block_intervals:
                tau_values = [int(block_intervals.get(block.block_id, 1)) for block in blocks]
                tau_min = min(tau_values)
                tau_max = max(tau_values)
                tau_norm = Normalize(vmin=tau_min, vmax=tau_max if tau_max > tau_min else tau_min + 1.0)
                tau_mappable = ScalarMappable(norm=tau_norm, cmap='plasma')

            for sub_row, frame_idx in enumerate(frame_indices):
                ax = axes[current_row, 0]
                frame = curvature_map[frame_idx].numpy()
                im = ax.imshow(frame, cmap='turbo', origin='upper', aspect='auto')
                ax.set_title(
                    f'Step {step_idx:04d} frame {frame_idx:03d} '
                    f'(curvature + {block_value_label} overlay)'
                )
                ax.set_xlabel('W token')
                ax.set_ylabel('H token')

                for block in blocks:
                    if not (block.f_start <= frame_idx < block.f_end):
                        continue

                    edgecolor = '#ffffff'
                    if tau_mappable is not None and block_intervals is not None:
                        tau = int(block_intervals.get(block.block_id, 1))
                        edgecolor = tau_mappable.to_rgba(tau)

                    rect = Rectangle(
                        (block.w_start, block.h_start),
                        block.w_size,
                        block.h_size,
                        fill=False,
                        linewidth=2.0,
                        edgecolor=edgecolor,
                    )
                    ax.add_patch(rect)

                    cx = block.w_start + block.w_size / 2.0
                    cy = block.h_start + block.h_size / 2.0
                    label = f'{block.f_size}x{block.h_size}x{block.w_size}'
                    if block_intervals is not None:
                        label = f'{block_value_label}={int(block_intervals.get(block.block_id, 1))}'
                    ax.text(
                        cx,
                        cy,
                        label,
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.55),
                    )

                cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.02)
                cbar.set_label('curvature')

                if tau_mappable is not None and sub_row == 0:
                    tau_cbar = fig.colorbar(tau_mappable, ax=ax, fraction=0.018, pad=0.09)
                    tau_cbar.set_label(block_value_label)

                current_row += 1

        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return last_frame_indices

"""

i1 = text.find("    def visualize_and_export(")
i2 = text.find("    def _build_summary_text(")
text_head = text[:i1]
text_middle = text[i2:]

i3 = text_middle.find("    def _save_frame_strip(")
i4 = text_middle.find("    def _format_stats(")
text_middle = text_middle[:i3] + text_middle[i4:]

text_final = text_head + new_methods + text_middle

i5 = text_final.find("    def export_curvature_history(")
i6 = text_final.find("    def export_straightness_records(")
export_curv = """    def export_curvature_history(
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

"""
text_final = text_final[:i5] + export_curv + text_final[i6:]

with open(f_path, 'w', encoding='utf-8') as f:
    f.write(text_final)
