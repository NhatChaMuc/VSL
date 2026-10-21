import torch
import torch.nn as nn
from torch import Tensor


class AttentionPooling(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attention = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        score = self.attention(x)
        score = torch.softmax(score, dim=1)
        return (score * x).sum(dim=1)


class TransformerClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 1605,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        hidden_dim: int = 512,
        num_classes: int = 29,
        dropout: float = 0.1,
        ff_dim: int | None = None,
    ) -> None:

        super().__init__()

        if ff_dim is not None:
            hidden_dim = ff_dim

        self.embedding = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.pooling = AttentionPooling(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.pooling(x)
        return self.classifier(x)