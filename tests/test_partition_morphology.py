from types import SimpleNamespace
import unittest

import torch

from cubic_core.partition import AdaptivePartitioner


class MorphologyPartitionTests(unittest.TestCase):
    def test_foreground_region_gets_finer_blocks(self):
        cfg = SimpleNamespace(
            partition_mode='morphology',
            min_block_size=2,
            max_block_size=8,
            min_block_tokens=1,
            max_block_tokens=1 << 30,
            video_spatial_max_tokens=1 << 30,
            cv_threshold=0.35,
            foreground_threshold_quantile=0.8,
            foreground_focus_block_size=4,
            foreground_mixed_ratio_low=0.15,
            foreground_mixed_ratio_high=0.85,
            morph_open_iters=1,
            morph_close_iters=1,
            morph_dilate_iters=1,
            morph_erode_iters=0,
            foreground_min_component_area=4,
        )
        partitioner = AdaptivePartitioner(cfg)

        curvature_map = torch.full((1, 16, 16), 0.1, dtype=torch.float32)
        curvature_map[:, 4:12, 5:11] = 2.0
        curvature_map[:, 6:10, 7:9] = 3.0

        partition = partitioner.partition(Ft=1, Ht=16, Wt=16, curvature_map=curvature_map)

        self.assertFalse(partition.is_uniform)
        self.assertEqual(partition.partition_mode, 'morphology')
        self.assertGreater(len(partition.blocks), 1)

        subject_tokens = []
        background_tokens = []
        for block in partition.blocks:
            center_h = 0.5 * (block.h_start + block.h_end)
            center_w = 0.5 * (block.w_start + block.w_end)
            if 4 <= center_h <= 12 and 5 <= center_w <= 11:
                subject_tokens.append(block.token_count)
            else:
                background_tokens.append(block.token_count)

        self.assertTrue(subject_tokens)
        self.assertTrue(background_tokens)
        self.assertLessEqual(max(subject_tokens), max(background_tokens))

    def test_uniform_mode_still_supported(self):
        cfg = SimpleNamespace(
            partition_mode='uniform',
            min_block_size=2,
            max_block_size=8,
            min_block_tokens=1,
            max_block_tokens=1 << 30,
            video_spatial_max_tokens=1 << 30,
            cv_threshold=0.5,
            max_blocksize_candidates=16,
        )
        partitioner = AdaptivePartitioner(cfg)
        curvature_map = torch.rand((1, 8, 8), dtype=torch.float32)

        partition = partitioner.partition(Ft=1, Ht=8, Wt=8, curvature_map=curvature_map)

        self.assertTrue(partition.is_uniform)
        self.assertEqual(partition.partition_mode, 'uniform')
        self.assertIsNotNone(partition.selected_block_size)


if __name__ == '__main__':
    unittest.main()
