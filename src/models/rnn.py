"""Recurrentes propios. Las salidas son z-scores; el adaptador revierte a kWh."""

from __future__ import annotations

import torch
from torch import nn


class LSTMMultiOutput(nn.Module):
    """Encoder LSTM y cabeza directa de H salidas con calendario futuro."""

    def __init__(self, n_features=8, n_future=7, H=24, hidden=32, layers=1):
        super().__init__()
        self.H = H
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + H * n_future, hidden), nn.ReLU(), nn.Linear(hidden, H)
        )

    def forward(self, x, x_fut):
        _, (h, _) = self.encoder(x)
        return self.head(torch.cat([h[-1], x_fut.flatten(1)], dim=1))


class LSTMAutoregressive(nn.Module):
    """Entrena un paso; en inferencia realimenta predicciones y calendario conocido."""

    def __init__(self, n_features=8, n_future=7, H=24, hidden=32, layers=1):
        super().__init__()
        self.H = H
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden + n_future, 1)

    def forward(self, x, x_fut):
        _, state = self.encoder(x)
        predictions = []
        for h in range(x_fut.shape[1]):
            y = self.head(torch.cat([state[0][-1], x_fut[:, h]], dim=-1))
            predictions.append(y)
            if h + 1 < x_fut.shape[1]:
                step = torch.cat([y, x_fut[:, h]], dim=-1).unsqueeze(1)
                _, state = self.encoder(step, state)
        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    """Encoder/decoder separados; teacher forcing exclusivamente al entrenar."""

    def __init__(
        self, n_features=8, n_future=7, H=24, hidden=32, layers=1, teacher_forcing=0.5
    ):
        super().__init__()
        self.H = H
        self.teacher_forcing = teacher_forcing
        self.encoder = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.decoder = nn.LSTM(1 + n_future, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def _decode(self, x, x_fut, targets=None):
        _, state = self.encoder(x)
        prev = x[:, -1, :1]
        out = []
        for h in range(x_fut.shape[1]):
            step = torch.cat([prev, x_fut[:, h]], dim=-1).unsqueeze(1)
            z, state = self.decoder(step, state)
            pred = self.head(z[:, 0])
            out.append(pred)
            if targets is not None and self.training:
                mask = torch.rand_like(pred) < self.teacher_forcing
                prev = torch.where(mask, targets[:, h : h + 1], pred)
            else:
                prev = pred
        return torch.cat(out, dim=1)

    def forward(self, x, x_fut):
        return self._decode(x, x_fut)

    def forward_train(self, x, x_fut, targets):
        return self._decode(x, x_fut, targets)
