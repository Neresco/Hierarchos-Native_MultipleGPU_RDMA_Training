"""Negative controls for the shipping trajectory comparison (no GPU needed)."""
import copy
import unittest

from verify_training_state import compare_snapshots


class TrainingStateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.expected = [{"step": 0, "optimizer_step": 1,
                          "optimizer_slot_steps": {"weight": 1},
                          "tensors": {"parameter/weight": "a", "exp_avg/weight": "b"}}]

    def test_exact_state_passes(self):
        self.assertEqual(compare_snapshots(self.expected, copy.deepcopy(self.expected)), [])

    def test_same_loss_cannot_hide_parameter_or_moment_drift(self):
        for tensor in self.expected[0]["tensors"]:
            observed = copy.deepcopy(self.expected)
            observed[0]["tensors"][tensor] = "changed"
            self.assertTrue(compare_snapshots(self.expected, observed))

    def test_missing_tensor_step_and_optimizer_clock_fail(self):
        observed = copy.deepcopy(self.expected)
        del observed[0]["tensors"]["parameter/weight"]
        self.assertTrue(compare_snapshots(self.expected, observed))
        self.assertTrue(compare_snapshots(self.expected, []))
        observed = copy.deepcopy(self.expected)
        observed[0]["optimizer_slot_steps"]["weight"] = 2
        self.assertTrue(compare_snapshots(self.expected, observed))


if __name__ == "__main__":
    unittest.main()
