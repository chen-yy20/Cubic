from types import SimpleNamespace
import unittest

import torch

from cubic_core.scheduler import SelectiveStepScheduler
from cubic_core.types import Block, Partition, Phase


class SelectiveStepSchedulerTests(unittest.TestCase):
    def _partition(self):
        return Partition(
            blocks=[
                Block(block_id=0, f_start=0, f_end=1, h_start=0, h_end=1, w_start=0, w_end=2),
                Block(block_id=1, f_start=0, f_end=1, h_start=1, h_end=2, w_start=0, w_end=2),
            ],
            Ft=1,
            Ht=2,
            Wt=2,
        )

    def test_selective_step_can_have_no_active_tokens(self):
        scheduler = SelectiveStepScheduler(SimpleNamespace())
        plan = scheduler.build_step_plan(
            partition=self._partition(),
            block_intervals={0: 2, 1: 3},
            phase=Phase.CRUISE_SELECTIVE,
            step_idx=1,
            selective_step_idx=1,
        )

        self.assertFalse(plan.is_full_compute)
        self.assertEqual(plan.active_token_indices.numel(), 0)
        self.assertEqual(plan.frozen_token_indices.numel(), 4)
        self.assertEqual(plan.kv_cache_token_indices.numel(), 0)

    def test_tau_one_tokens_are_active_but_skip_persistent_kv_cache_updates(self):
        scheduler = SelectiveStepScheduler(SimpleNamespace())
        plan = scheduler.build_step_plan(
            partition=self._partition(),
            block_intervals={0: 1, 1: 2},
            phase=Phase.CRUISE_SELECTIVE,
            step_idx=1,
            selective_step_idx=1,
        )

        self.assertFalse(plan.is_full_compute)
        self.assertTrue(torch.equal(plan.active_token_indices, torch.tensor([0, 1])))
        self.assertEqual(plan.kv_cache_token_indices.numel(), 0)

        plan = scheduler.build_step_plan(
            partition=self._partition(),
            block_intervals={0: 1, 1: 2},
            phase=Phase.CRUISE_SELECTIVE,
            step_idx=2,
            selective_step_idx=2,
        )

        self.assertTrue(torch.equal(plan.active_token_indices, torch.tensor([0, 1, 2, 3])))
        self.assertTrue(plan.is_full_compute)


if __name__ == '__main__':
    unittest.main()
