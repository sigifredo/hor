'''WaveNet causal dilatado.

Dos modos de cómputo sobre los mismos pesos:
  - WaveNet.forward: paralelo, para entrenamiento con teacher forcing. Las
    convoluciones dilatadas no tienen dependencia secuencial en entrenamiento,
    así que un chunk completo se procesa en una sola pasada.
  - WaveNetGenerator.step: incremental, muestra a muestra, con caché de
    activaciones por capa (fast-wavenet). Genera cada muestra reutilizando el
    cómputo previo en vez de recomputar el campo receptivo entero.
'''

import copy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .mu_law import SILENCE_INDEX


class WaveNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.residual_channels = cfg.residual_channels
        self.dilation_channels = cfg.dilation_channels
        self.skip_channels = cfg.skip_channels
        self.quant_levels = cfg.quant_levels
        self.kernel_size = cfg.kernel_size
        self.dilations = cfg.dilations

        self.embed = nn.Embedding(cfg.quant_levels, cfg.residual_channels)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for d in self.dilations:
            self.filter_convs.append(
                nn.Conv1d(cfg.residual_channels, cfg.dilation_channels,
                          cfg.kernel_size, dilation=d))
            self.gate_convs.append(
                nn.Conv1d(cfg.residual_channels, cfg.dilation_channels,
                          cfg.kernel_size, dilation=d))
            self.residual_convs.append(
                nn.Conv1d(cfg.dilation_channels, cfg.residual_channels, 1))
            self.skip_convs.append(
                nn.Conv1d(cfg.dilation_channels, cfg.skip_channels, 1))

        self.out_conv1 = nn.Conv1d(cfg.skip_channels, cfg.skip_channels, 1)
        self.out_conv2 = nn.Conv1d(cfg.skip_channels, cfg.quant_levels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: (B, T) índices long. Devuelve logits (B, quant_levels, T).
        El logit en la posición t predice la muestra t+1.'''
        h = self.embed(x).transpose(1, 2)            # (B, C, T)
        skip_total = 0
        for i, d in enumerate(self.dilations):
            pad = d * (self.kernel_size - 1)
            hp = F.pad(h, (pad, 0))                   # padding causal a la izquierda
            f = torch.tanh(self.filter_convs[i](hp))
            g = torch.sigmoid(self.gate_convs[i](hp))
            z = f * g
            skip_total = skip_total + self.skip_convs[i](z)
            h = h + self.residual_convs[i](z)
        out = F.relu(skip_total)
        out = F.relu(self.out_conv1(out))
        return self.out_conv2(out)                    # (B, quant_levels, T)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def clone_for_inference(model: WaveNet) -> WaveNet:
    '''Copia congelada e independiente para el hilo de generación.
    Ejecuta en el hilo de entrenamiento, fuera de la ruta de tiempo real.'''
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class WaveNetGenerator:
    '''Generación autoregresiva incremental con caché por capa.

    Mantiene una FIFO por capa de longitud igual a su dilatación, que almacena
    las entradas pasadas a esa capa. Con kernel 2, la conv dilatada en el paso t
    solo necesita la entrada t y la entrada t-d, así que la FIFO basta.

    El puntero `model` puede reasignarse en caliente (swap atómico de pesos).
    Las cachés persisten a través del swap: cada commit introduce un transitorio
    perceptible, tratado como material sonoro, no como artefacto a suprimir.'''

    def __init__(self, model: WaveNet):
        self.model = model
        self.dilations = model.dilations
        self.residual_channels = model.residual_channels
        self.skip_channels = model.skip_channels
        self.last_idx = SILENCE_INDEX
        self.reset_caches()

    def reset_caches(self):
        self.queues = [
            deque([torch.zeros(self.residual_channels) for _ in range(d)],
                  maxlen=d)
            for d in self.dilations
        ]
        self.last_idx = SILENCE_INDEX

    @torch.no_grad()
    def step(self, temperature: float = 1.0) -> int:
        '''Genera un índice mu-law y actualiza el estado interno.'''
        m = self.model
        idx = torch.tensor([self.last_idx], dtype=torch.long)
        h = m.embed(idx).squeeze(0)                   # (residual_channels,)
        skip_total = torch.zeros(self.skip_channels)

        for i, d in enumerate(self.dilations):
            past = self.queues[i][0]                  # entrada t-d (más antigua)
            fw = m.filter_convs[i].weight             # (dil, res, 2)
            gw = m.gate_convs[i].weight
            f = torch.tanh(fw[:, :, 0] @ past + fw[:, :, 1] @ h
                           + m.filter_convs[i].bias)
            g = torch.sigmoid(gw[:, :, 0] @ past + gw[:, :, 1] @ h
                              + m.gate_convs[i].bias)
            z = f * g                                 # (dilation_channels,)

            sw = m.skip_convs[i].weight               # (skip, dil, 1)
            skip_total = skip_total + sw[:, :, 0] @ z + m.skip_convs[i].bias
            rw = m.residual_convs[i].weight           # (res, dil, 1)
            res = rw[:, :, 0] @ z + m.residual_convs[i].bias

            self.queues[i].append(h)                  # empuja entrada t
            h = h + res                               # residual -> siguiente capa

        out = torch.relu(skip_total)
        o1 = m.out_conv1.weight
        out = torch.relu(o1[:, :, 0] @ out + m.out_conv1.bias)
        o2 = m.out_conv2.weight
        logits = o2[:, :, 0] @ out + m.out_conv2.bias  # (quant_levels,)

        t = max(temperature, 1e-3)
        probs = torch.softmax(logits / t, dim=0)
        sampled = int(torch.multinomial(probs, 1).item())
        self.last_idx = sampled
        return sampled
