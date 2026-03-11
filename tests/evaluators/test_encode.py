from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from popgen.population.encode import TextEncoder, TrajectoryEncoder
from popgen.utils.config import LitellmConfig


def _mock_embedding_response(vectors: list[list[float]]) -> MagicMock:
    response = MagicMock()
    response.data = [{"embedding": vec} for vec in vectors]
    return response


def test_text_encoder_batch_encoder():
    encoder = TextEncoder(embedding_model=LitellmConfig(model_name="dummy-model"), batch_size=2)
    with patch("popgen.population.encode.litellm.embedding") as mock_embedding:
        mock_embedding.return_value = _mock_embedding_response([[1.0, 0.0], [0.0, 1.0]])
        out = encoder.batch_encoder(["a", "b"])
    assert out.shape == (2, 2)
    assert np.allclose(out[0], np.array([1.0, 0.0]))


class _DummyTrajectoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_actions = 3

    def forward(self, states: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del actions
        pooled = (states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


def test_trajectory_encoder_batch_encoder():
    model = _DummyTrajectoryModel()
    encoder = TrajectoryEncoder(model=model, device="cpu")
    out = encoder.batch_encoder(
        [
            {"states": [[0.0, 0.0], [1.0, 0.0]], "actions": [0, 1]},
            {"states": [[0.0, 1.0], [0.0, 2.0]], "actions": [1, 1]},
        ]
    )
    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
