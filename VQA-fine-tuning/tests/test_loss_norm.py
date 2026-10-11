import unittest
import torch
from vrsbench_finetune.train import compute_loss

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('obj', (object,), {'pad_token_id': 0})
        
    def forward(self, input_ids, attention_mask, pixel_values, labels):
        # Return dummy logits
        batch_size, seq_len = input_ids.shape
        vocab_size = 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return type('obj', (object,), {'logits': logits})

class TestLossNorm(unittest.TestCase):
    def test_loss_calculation(self):
        model = MockModel()
        batch = {
            "input_ids": torch.randint(0, 100, (2, 10)),
            "labels": torch.randint(0, 100, (2, 10)),
            "tasks": ["binary", "numeric"]
        }
        # Mask some labels
        batch["labels"][:, :5] = -100
        
        task_weights = {"binary": 1.0, "numeric": 1.0}
        
        loss = compute_loss(model, batch, task_weights)
        self.assertTrue(isinstance(loss, torch.Tensor))
        self.assertFalse(torch.isnan(loss))

if __name__ == '__main__':
    unittest.main()
