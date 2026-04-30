from types import SimpleNamespace
import unittest

from cubic_core.optimizer import UpdateFrequencyOptimizer
from cubic_core.types import Block, Partition


class WeightedSpeedupOptimizerTests(unittest.TestCase):
    def test_target_speedup_uses_token_weighted_rate(self):
        cfg = SimpleNamespace(
            alpha=2.0,
            beta=4.0,
            target_speedup=3.0,
            curvature_contrast_gamma=1.0,
        )
        optimizer = UpdateFrequencyOptimizer(cfg)
        partition = Partition(
            blocks=[
                Block(block_id=0, f_start=0, f_end=1, h_start=0, h_end=9, w_start=0, w_end=10),
                Block(block_id=1, f_start=0, f_end=1, h_start=9, h_end=10, w_start=0, w_end=10),
            ],
            Ft=1,
            Ht=10,
            Wt=10,
        )
        block_curvatures = {
            0: 1.0,
            1: 100.0,
        }

        plan = optimizer.optimize(partition, block_curvatures)

        tokens = [block.token_count for block in partition.blocks]
        taus = [plan.block_intervals[block.block_id] for block in partition.blocks]
        weighted_rate = sum(token / tau for token, tau in zip(tokens, taus)) / sum(tokens)
        block_rate = sum(1.0 / tau for tau in taus) / len(taus)

        self.assertAlmostEqual(plan.mean_update_rate, weighted_rate, places=8)
        self.assertAlmostEqual(plan.achieved_speedup, 1.0 / weighted_rate, places=8)
        self.assertAlmostEqual(plan.block_mean_update_rate, block_rate, places=8)
        self.assertAlmostEqual(plan.block_achieved_speedup, 1.0 / block_rate, places=8)
        self.assertLess(abs(weighted_rate - 1.0 / 3.0), abs(block_rate - 1.0 / 3.0))

    def test_higher_curvature_blocks_get_no_larger_tau(self):
        cfg = SimpleNamespace(
            alpha=2.0,
            beta=4.0,
            target_speedup=3.0,
            curvature_contrast_gamma=1.0,
        )
        optimizer = UpdateFrequencyOptimizer(cfg)
        partition = Partition(
            blocks=[
                Block(block_id=0, f_start=0, f_end=1, h_start=0, h_end=1, w_start=0, w_end=40),
                Block(block_id=1, f_start=0, f_end=1, h_start=1, h_end=2, w_start=0, w_end=40),
                Block(block_id=2, f_start=0, f_end=1, h_start=2, h_end=3, w_start=0, w_end=40),
            ],
            Ft=1,
            Ht=3,
            Wt=40,
        )
        block_curvatures = {
            0: 1.0,
            1: 10.0,
            2: 100.0,
        }

        plan = optimizer.optimize(partition, block_curvatures)
        taus = [plan.block_intervals[i] for i in range(3)]

        self.assertGreaterEqual(taus[0], taus[1])
        self.assertGreaterEqual(taus[1], taus[2])


if __name__ == '__main__':
    unittest.main()
