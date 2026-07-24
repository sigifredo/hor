'''WaveNet causal dilatado con salida DMoL (Discretized Mixture of Logistics).

Cambios frente a la versión mu-law categórica:
  - Entrada: señal continua en [-1, 1] (float32), no índices categóricos.
    El embedding pasa de `nn.Embedding(256, res)` a `nn.Conv1d(1, res, 1)`.
  - Salida: 3*K parámetros (pi, mu, log_s) por muestra en vez de 256 logits.
  - Muestreo: se elige componente, se muestrea de logística, se cuantiza a 16 bits.

Dos rutas de cómputo:
  - WaveNet.forward (torch, paralelo): entrenamiento con teacher forcing.
  - WaveNetGenerator (torch, incremental por muestra): generación autoregresiva
    en CPU o GPU. Con salida DMoL desapareció la ventaja del generador NumPy
    optimizado (más simple mantener una sola ruta en torch), pero sigue siendo
    más lento que el modo paralelo por overhead de dispatch por muestra.
'''

from .config import Config
from . import dmol

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_mix = cfg.n_mix
        self.kernel_size = cfg.kernel_size
        self.dilations = cfg.dilations

        # Entrada continua: 1 canal (amplitud) -> residual_channels.
        # 1x1 conv equivale a linear per-step; usa Conv1d por consistencia con el resto.
        self.embed = nn.Conv1d(1, cfg.residual_channels, 1)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for d in self.dilations:
            self.filter_convs.append(nn.Conv1d(cfg.residual_channels, cfg.dilation_channels, cfg.kernel_size, dilation=d))
            self.gate_convs.append(nn.Conv1d(cfg.residual_channels, cfg.dilation_channels, cfg.kernel_size, dilation=d))
            self.residual_convs.append(nn.Conv1d(cfg.dilation_channels, cfg.residual_channels, 1))
            self.skip_convs.append(nn.Conv1d(cfg.dilation_channels, cfg.skip_channels, 1))

        self.out_conv1 = nn.Conv1d(cfg.skip_channels, cfg.skip_channels, 1)
        self.out_conv2 = nn.Conv1d(cfg.skip_channels, dmol.n_params(cfg.n_mix), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: (B, T) float32 en [-1, 1]. Devuelve params (B, 3*K, T).
        Los params en la posición t predicen la muestra t+1.'''
        h = self.embed(x.unsqueeze(1))  # (B, res, T)
        skip_total = 0

        for i, d in enumerate(self.dilations):
            pad = d * (self.kernel_size - 1)
            hp = F.pad(h, (pad, 0))
            f = torch.tanh(self.filter_convs[i](hp))
            g = torch.sigmoid(self.gate_convs[i](hp))
            z = f * g
            skip_total = skip_total + self.skip_convs[i](z)
            h = h + self.residual_convs[i](z)

        out = F.relu(skip_total)
        out = F.relu(self.out_conv1(out))
        return self.out_conv2(out)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def cpu_state_dict(model: WaveNet) -> dict:
    '''Instantánea de pesos en CPU, fuera del grafo de autograd.'''
    return {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}


class WaveNetGenerator:
    '''Generación autoregresiva incremental en torch (CPU o GPU).

    Con salida DMoL el bucle interno de generación es esencialmente equivalente al
    forward paralelo aplicado a un tensor de longitud 1 con ring buffers por capa,
    así que reusar directamente los módulos de torch simplifica el mantenimiento:
    los pesos se leen directamente del state_dict con load_state_dict, y el forward
    incremental replica la lógica del forward paralelo respetando el ring buffer.

    Mantiene la interfaz de la versión NumPy (reset_caches, load_state, generate,
    step, last_value, version) para minimizar cambios en engine.py.'''

    def __init__(self, cfg: Config, device: str = 'cpu'):
        assert cfg.kernel_size == 2, 'el generador incremental asume kernel 2'
        self.cfg = cfg
        self.device = torch.device(device)
        self.n_mix = cfg.n_mix
        self.dilations = list(cfg.dilations)
        self.version = -1

        # Estado incremental: ring buffers por capa, un valor de "última muestra" continuo.
        self.model = WaveNet(cfg).to(self.device)
        self.model.eval()
        self.rings = [torch.zeros(d, cfg.residual_channels, device=self.device) for d in self.dilations]
        self.ptrs = [0] * len(self.dilations)
        self.last_value = 0.0  # arranca en silencio (0.0 en [-1, 1])

    def reset_caches(self) -> None:
        for r in self.rings:
            r.zero_()
        self.ptrs = [0] * len(self.dilations)
        self.last_value = 0.0

    def load_state(self, state: dict, version: int | None = None) -> None:
        '''Carga pesos desde un state_dict. Cachés persisten.'''
        # Mover state al device del modelo antes de load
        state_dev = {k: v.to(self.device) for k, v in state.items()}
        self.model.load_state_dict(state_dev)
        if version is not None:
            self.version = version

    @torch.no_grad()
    def generate(self, n: int, temperature: float = 1.0) -> torch.Tensor:
        '''Genera n muestras (float32 en [-1, 1] cuantizadas a 16 bits).'''
        out = torch.empty(n, dtype=torch.float32, device=self.device)
        m = self.model
        for j in range(n):
            # Embed la última muestra: (1, 1, 1) -> (1, res, 1)
            x_in = torch.tensor([[[self.last_value]]], device=self.device)
            h = m.embed(x_in).squeeze(0).squeeze(-1)  # (res,)
            skip_total = torch.zeros(m.skip_convs[0].out_channels, device=self.device)

            for i, d in enumerate(self.dilations):
                ring = self.rings[i]
                p = self.ptrs[i]
                past = ring[p]  # (res,)
                cur = h  # (res,)

                # Conv de kernel 2: sale = W_past @ past + W_cur @ cur + bias.
                # Los pesos de nn.Conv1d con kernel=2 tienen forma (out, in, 2).
                fw = m.filter_convs[i].weight  # (dil, res, 2)
                fb = m.filter_convs[i].bias
                gw = m.gate_convs[i].weight
                gb = m.gate_convs[i].bias

                f = torch.tanh(fw[:, :, 0] @ past + fw[:, :, 1] @ cur + fb)
                g = torch.sigmoid(gw[:, :, 0] @ past + gw[:, :, 1] @ cur + gb)
                z = f * g

                sw = m.skip_convs[i].weight.squeeze(-1)  # (skip, dil)
                sb = m.skip_convs[i].bias
                rw = m.residual_convs[i].weight.squeeze(-1)  # (res, dil)
                rb = m.residual_convs[i].bias

                skip_total = skip_total + sw @ z + sb
                # Empuja h al ring antes de actualizar h con el residual
                ring[p] = h
                self.ptrs[i] = (p + 1) % d
                h = h + rw @ z + rb

            o = F.relu(skip_total)
            o = F.relu(m.out_conv1.weight.squeeze(-1) @ o + m.out_conv1.bias)
            params = m.out_conv2.weight.squeeze(-1) @ o + m.out_conv2.bias  # (3K,)
            # Reformatear a (1, 3K, 1) para reusar dmol_sample
            params = params.view(1, -1, 1)
            sample = dmol.dmol_sample(params, self.n_mix, temperature)  # (1, 1)
            v = float(sample.item())
            self.last_value = v
            out[j] = v

        return out

    def step(self, temperature: float = 1.0) -> float:
        '''Compatibilidad: genera un solo valor.'''
        return float(self.generate(1, temperature).item())
