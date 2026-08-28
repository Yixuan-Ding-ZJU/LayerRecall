import unittest

import torch

from utils.layer_recall import (
    materialize_layer_recall_slot,
    straight_through_hard_value,
)


class YXStraightThroughTest(unittest.TestCase):
    def test_hard_and_soft_inference_materialize_different_values(self):
        hard = torch.tensor([1.0, 2.0])
        soft = torch.tensor([3.0, 4.0])

        self.assertIs(materialize_layer_recall_slot(hard, soft, "hard"), hard)
        self.assertIs(materialize_layer_recall_slot(hard, soft, "soft"), soft)

    def test_selection_mode_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported LayerRecall"):
            materialize_layer_recall_slot(
                torch.zeros(2),
                torch.zeros(2),
                "unknown",
            )

    def test_forward_is_bitwise_hard_in_bfloat16(self):
        torch.manual_seed(0)
        hard = torch.randn(4096, dtype=torch.bfloat16)
        soft = (10 * torch.randn_like(hard)).requires_grad_(True)

        result = straight_through_hard_value(hard, soft)

        self.assertTrue(torch.equal(result, hard))

    def test_routes_gradient_through_soft_value(self):
        hard = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        soft = torch.randn(4, 5, dtype=torch.float32, requires_grad=True)
        upstream = torch.randn(4, 5, dtype=torch.float32)

        result = straight_through_hard_value(hard, soft)
        hard_grad, soft_grad = torch.autograd.grad(
            (result * upstream).sum(),
            (hard, soft),
        )

        self.assertTrue(torch.equal(hard_grad, upstream))
        self.assertTrue(torch.equal(soft_grad, upstream))

    def test_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            straight_through_hard_value(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
