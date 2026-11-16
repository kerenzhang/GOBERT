"""Small regression checks for sparse graph training and checkpoint loading."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig

from gobert import GoBERTModel_Ex_Im


class SparseTrainingTests(unittest.TestCase):
    def make_model(self):
        config = BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=16, max_position_embeddings=16)
        graph = torch.eye(8)
        graph[1, 2] = 1
        with patch.object(torch, "load", return_value=graph.to_sparse()), patch.object(np, "load", return_value=np.ones((3, 8), dtype=np.float32)):
            model = GoBERTModel_Ex_Im(config, semantic_path="mock.npy")
        return model, graph

    def test_sparse_rows_match_dense_with_repeated_tokens(self):
        model, graph = self.make_model()
        ids = torch.tensor([[1, 1, 3], [2, 4, 1]])
        rows = torch.index_select(model.ex_label_buffer, 0, ids.reshape(-1)).to_dense()
        torch.testing.assert_close(rows, graph[ids].reshape(-1, 8))
        self.assertEqual(model.ex_label_buffer.layout, torch.sparse_coo)
        self.assertNotIn("ex_label_buffer", model.state_dict())
        self.assertIs(model.bert.embeddings.word_embeddings.weight, model.cls.predictions.decoder.weight)

    def test_backward_and_sparse_checkpoint_reload(self):
        model, _ = self.make_model()
        batch = {"input_ids": torch.tensor([[1, 2, 3]]), "labels": torch.tensor([[-100, 2, -100]])}
        result = model(**batch)
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()
        self.assertIsNotNone(model.bert.embeddings.word_embeddings.weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=False)
            restored, _ = self.make_model()
            restored.load_state_dict(torch.load(Path(directory) / "pytorch_model.bin", weights_only=True))
            torch.testing.assert_close(restored.ex_label_buffer.to_dense(), model.ex_label_buffer.to_dense())
            torch.testing.assert_close(restored.bert.embeddings.word_embeddings.weight, model.bert.embeddings.word_embeddings.weight)

    def test_no_masked_tokens_has_finite_loss(self):
        model, _ = self.make_model()
        result = model(input_ids=torch.tensor([[1, 2, 3]]), labels=torch.full((1, 3), -100))
        self.assertTrue(torch.isfinite(result["loss"]))
        result["loss"].backward()


if __name__ == "__main__":
    unittest.main()
