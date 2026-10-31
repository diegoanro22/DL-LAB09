"""Modelos propios en PyTorch: MLP multi-salida, DLinear y NLinear (sección 5).

Todos reciben x: (B, L, F) del WindowDataset (columna 0 = y normalizada por serie,
columnas 1.. = calendario) y devuelven (B, H): estrategia multi-salida (los 24 pasos de
una vez). El MLP además puede recibir x_fut: (B, H, C), el calendario del horizonte.
"""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Reversible Instance Normalization (Kim et al., 2022).

    Normaliza cada ventana con su propia media y desviación (solo la columna objetivo)
    y revierte la salida con los mismos estadísticos. Ayuda cuando el nivel de una serie
    cambia respecto al de entrenamiento: el modelo aprende la forma, no el nivel.
    """

    def __init__(self, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))

    def norm(self, y: torch.Tensor) -> torch.Tensor:          # y: (B, L)
        self.mu = y.mean(1, keepdim=True).detach()
        self.sigma = (y.var(1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        out = (y - self.mu) / self.sigma
        return out * self.gamma + self.beta if self.affine else out

    def denorm(self, out: torch.Tensor) -> torch.Tensor:      # out: (B, H)
        if self.affine:
            out = (out - self.beta) / (self.gamma + self.eps)
        return out * self.sigma + self.mu


class NLinear(nn.Module):
    """NLinear (Zeng et al., 2023): resta el último valor, capa lineal L -> H, lo suma de vuelta.

    Solo usa la serie objetivo (univariado, compartido entre todas las series).
    """

    def __init__(self, L: int, H: int, revin: bool = False):
        super().__init__()
        self.linear = nn.Linear(L, H)
        self.revin = RevIN() if revin else None

    def forward(self, x, x_fut=None):
        y = x[..., 0]
        if self.revin is not None:
            y = self.revin.norm(y)
        last = y[:, -1:]
        out = self.linear(y - last) + last
        return self.revin.denorm(out) if self.revin is not None else out


class MovingAvg(nn.Module):
    """Media móvil con relleno replicado en los bordes (mantiene el largo L)."""

    def __init__(self, kernel: int):
        super().__init__()
        self.kernel = kernel
        self.pool = nn.AvgPool1d(kernel, stride=1)

    def forward(self, y):                                     # (B, L)
        left = (self.kernel - 1) // 2
        right = self.kernel - 1 - left
        y = torch.cat([y[:, :1].repeat(1, left), y, y[:, -1:].repeat(1, right)], dim=1)
        return self.pool(y.unsqueeze(1)).squeeze(1)


class DLinear(nn.Module):
    """DLinear (Zeng et al., 2023): descompone en tendencia (media móvil) + resto y aplica
    una capa lineal L -> H a cada parte; el pronóstico es la suma."""

    def __init__(self, L: int, H: int, kernel: int = 25, revin: bool = False):
        super().__init__()
        self.decomp = MovingAvg(kernel)
        self.lin_trend = nn.Linear(L, H)
        self.lin_season = nn.Linear(L, H)
        self.revin = RevIN() if revin else None

    def forward(self, x, x_fut=None):
        y = x[..., 0]
        if self.revin is not None:
            y = self.revin.norm(y)
        trend = self.decomp(y)
        out = self.lin_trend(trend) + self.lin_season(y - trend)
        return self.revin.denorm(out) if self.revin is not None else out


class MLP(nn.Module):
    """MLP multi-salida: aplana la ventana (L x F) y, si hay, el calendario del horizonte
    (H x C), y devuelve los H pasos con una capa densa final."""

    def __init__(self, L: int, H: int, n_features: int, n_future: int = 0,
                 hidden: int = 512, layers: int = 2, dropout: float = 0.1, revin: bool = False):
        super().__init__()
        d_in = L * n_features + H * n_future
        blocks, d = [], d_in
        for _ in range(layers):
            blocks += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(d, H)
        self.revin = RevIN() if revin else None
        self.n_future = n_future

    def forward(self, x, x_fut=None):
        if self.revin is not None:
            y = self.revin.norm(x[..., 0])
            x = torch.cat([y.unsqueeze(-1), x[..., 1:]], dim=-1)
        parts = [x.flatten(1)]
        if self.n_future:
            parts.append(x_fut.flatten(1))
        out = self.head(self.body(torch.cat(parts, dim=1)))
        return self.revin.denorm(out) if self.revin is not None else out
