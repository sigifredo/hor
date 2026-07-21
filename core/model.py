'''WaveNet causal dilatado.

Dos rutas de cómputo sobre los mismos pesos:
  - WaveNet.forward (torch): paralelo, para entrenamiento con teacher forcing.
    Las convoluciones dilatadas no tienen dependencia secuencial en
    entrenamiento, así que un chunk completo se procesa en una sola pasada.
  - WaveNetGenerator (numpy, CPU): incremental, muestra a muestra, con caché de
    activaciones por capa (fast-wavenet) y pesos fusionados.

La generación autoregresiva es secuencial por naturaleza: no se puede
paralelizar sobre el tiempo. En GPU cada muestra paga ~50 lanzamientos de
kernel más una sincronización GPU->CPU (el muestreo necesita el índice en
Python), y ese overhead fijo domina por completo el cómputo de un modelo de
este tamaño (matrices 64x64). Por eso el generador vive en CPU y en NumPy:
matvecs pequeños sin dispatch de framework.
'''

from .config import Config
from .mu_law import SILENCE_INDEX

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.filter_convs.append(nn.Conv1d(cfg.residual_channels, cfg.dilation_channels, cfg.kernel_size, dilation=d))
            self.gate_convs.append(nn.Conv1d(cfg.residual_channels, cfg.dilation_channels, cfg.kernel_size, dilation=d))
            self.residual_convs.append(nn.Conv1d(cfg.dilation_channels, cfg.residual_channels, 1))
            self.skip_convs.append(nn.Conv1d(cfg.dilation_channels, cfg.skip_channels, 1))

        self.out_conv1 = nn.Conv1d(cfg.skip_channels, cfg.skip_channels, 1)
        self.out_conv2 = nn.Conv1d(cfg.skip_channels, cfg.quant_levels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''x: (B, T) índices long. Devuelve logits (B, quant_levels, T).
        El logit en la posición t predice la muestra t+1.'''
        h = self.embed(x).transpose(1, 2)  # (B, C, T)
        skip_total = 0

        for i, d in enumerate(self.dilations):
            pad = d * (self.kernel_size - 1)
            hp = F.pad(h, (pad, 0))  # padding causal a la izquierda
            f = torch.tanh(self.filter_convs[i](hp))
            g = torch.sigmoid(self.gate_convs[i](hp))
            z = f * g
            skip_total = skip_total + self.skip_convs[i](z)
            h = h + self.residual_convs[i](z)

        out = F.relu(skip_total)
        out = F.relu(self.out_conv1(out))
        return self.out_conv2(out)  # (B, quant_levels, T)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def cpu_state_dict(model: WaveNet) -> dict:
    '''Instantánea de pesos en CPU, fuera del grafo de autograd.

    Reemplaza el deepcopy del modelo completo: publicar pesos al generador es
    copiar ~1.4 MB de tensores, no clonar un nn.Module. Ejecuta en el hilo de
    entrenamiento, fuera de la ruta de tiempo real.'''
    return {k: v.detach().to('cpu', copy=True) for k, v in model.state_dict().items()}


class WaveNetGenerator:
    '''Generación autoregresiva incremental en NumPy puro (CPU).

    Por capa mantiene un ring buffer de longitud d con las entradas pasadas a
    esa capa. Con kernel 2, la conv dilatada en el paso t solo necesita la
    entrada t y la entrada t-d, así que el ring basta.

    Los pesos llegan por load_state() (instantánea CPU de state_dict) y se
    fusionan una sola vez por publicación:
      - filtro y compuerta, sobre {past, cur}, en una matriz (2*dil, 2*res)
      - skip y residual en una matriz (skip+res, dil)
    Resultado: 2 matvecs por capa + 2 de salida por muestra, sin dispatch de
    framework. El muestreo usa Gumbel-max (equivalente exacto a multinomial
    sobre softmax(logits/T)) con ruido pregenerado por bloques.

    Las cachés persisten a través de cada load_state: cada publicación de
    pesos introduce un transitorio perceptible, tratado como material sonoro,
    no como artefacto a suprimir. `version` permite al hilo de generación
    recargar pesos solo cuando realmente cambiaron.'''

    def __init__(self, cfg: Config):
        assert cfg.kernel_size == 2, 'el generador incremental asume kernel 2'
        self.dilations = list(cfg.dilations)
        self.res = cfg.residual_channels
        self.dil = cfg.dilation_channels
        self.skip = cfg.skip_channels
        self.quant = cfg.quant_levels
        self.version = -1
        self.last_idx = SILENCE_INDEX
        self.rings = [np.zeros((d, self.res), dtype=np.float32) for d in self.dilations]
        self.ptrs = [0] * len(self.dilations)

    def reset_caches(self) -> None:
        for r in self.rings:
            r.fill(0.0)
        self.ptrs = [0] * len(self.dilations)
        self.last_idx = SILENCE_INDEX

    def load_state(self, state: dict, version: int | None = None) -> None:
        '''Extrae y fusiona pesos desde un state_dict en CPU. Las cachés no se
        tocan (continuidad sonora a través del swap).'''

        def f32(key):
            return state[key].detach().cpu().numpy().astype(np.float32, copy=True)

        self.embed_w = np.ascontiguousarray(f32('embed.weight'))  # (Q, res)
        self.W_fg, self.b_fg, self.W_sr, self.b_sr = [], [], [], []

        for i in range(len(self.dilations)):
            fw = f32(f'filter_convs.{i}.weight')  # (dil, res, 2): [:, :, 0]=t-d, [:, :, 1]=t
            gw = f32(f'gate_convs.{i}.weight')
            # Columnas: [past | cur]; filas: [filtro | compuerta].
            top = np.concatenate((fw[:, :, 0], fw[:, :, 1]), axis=1)
            # Sigmoide vía tanh: sig(x) = 0.5*(1+tanh(0.5*x)). El 0.5 interior
            # se pliega aquí en pesos y sesgo de compuerta, de modo que en
            # generate() basta UN tanh sobre el vector [filtro|compuerta].
            bot = 0.5 * np.concatenate((gw[:, :, 0], gw[:, :, 1]), axis=1)
            self.W_fg.append(np.ascontiguousarray(np.concatenate((top, bot), axis=0)))
            self.b_fg.append(np.concatenate((f32(f'filter_convs.{i}.bias'), 0.5 * f32(f'gate_convs.{i}.bias'))))

            sw = f32(f'skip_convs.{i}.weight')[:, :, 0]  # (skip, dil)
            rw = f32(f'residual_convs.{i}.weight')[:, :, 0]  # (res, dil)
            self.W_sr.append(np.ascontiguousarray(np.concatenate((sw, rw), axis=0)))
            self.b_sr.append(np.concatenate((f32(f'skip_convs.{i}.bias'), f32(f'residual_convs.{i}.bias'))))

        self.W_o1 = np.ascontiguousarray(f32('out_conv1.weight')[:, :, 0])
        self.b_o1 = f32('out_conv1.bias')
        self.W_o2 = np.ascontiguousarray(f32('out_conv2.weight')[:, :, 0])
        self.b_o2 = f32('out_conv2.bias')

        if version is not None:
            self.version = version

    def generate(self, n: int, temperature: float = 1.0) -> np.ndarray:
        '''Genera n índices mu-law (int64) actualizando el estado interno.'''
        out = np.empty(n, dtype=np.int64)
        inv_t = np.float32(1.0 / max(temperature, 1e-3))
        S = self.skip
        D = self.dil
        L = len(self.dilations)
        # Buffers reutilizados: el costo por muestra lo domina el overhead de
        # intérprete + despacho de numpy, así que se minimizan llamadas y allocs.
        pc = np.empty(2 * self.res, dtype=np.float32)  # [past | cur]
        fg = np.empty(2 * D, dtype=np.float32)
        sr = np.empty(S + self.res, dtype=np.float32)
        noise = np.empty((0, self.quant), dtype=np.float32)
        noise_i = 0

        for j in range(n):
            if noise_i >= len(noise):  # ruido Gumbel por bloques (amortiza el RNG)
                noise = np.random.gumbel(size=(min(2048, n - j), self.quant)).astype(np.float32)
                noise_i = 0

            h = self.embed_w[self.last_idx]
            skip_total = np.zeros(S, dtype=np.float32)

            for i in range(L):
                ring = self.rings[i]
                p = self.ptrs[i]
                pc[: self.res] = ring[p]  # entrada t-d (más antigua)
                pc[self.res :] = h
                np.matmul(self.W_fg[i], pc, out=fg)
                fg += self.b_fg[i]
                np.tanh(fg, out=fg)  # compuerta ya escalada 0.5 en load_state
                z = fg[:D] * (0.5 + 0.5 * fg[D:])  # tanh(f) * sigmoide(g)

                np.matmul(self.W_sr[i], z, out=sr)
                sr += self.b_sr[i]
                skip_total += sr[:S]

                ring[p] = h  # empuja entrada t (sobrescribe la más antigua)
                self.ptrs[i] = (p + 1) % self.dilations[i]
                h = h + sr[S:]  # residual -> siguiente capa

            np.maximum(skip_total, 0.0, out=skip_total)
            o = self.W_o1 @ skip_total
            o += self.b_o1
            np.maximum(o, 0.0, out=o)
            logits = self.W_o2 @ o
            logits += self.b_o2

            # Gumbel-max == muestrear de softmax(logits / T).
            idx = int(np.argmax(logits * inv_t + noise[noise_i]))
            noise_i += 1
            self.last_idx = idx
            out[j] = idx

        return out

    def step(self, temperature: float = 1.0) -> int:
        '''Compatibilidad: genera un solo índice.'''
        return int(self.generate(1, temperature)[0])
