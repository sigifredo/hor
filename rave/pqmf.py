'''Banco de filtros PQMF para análisis y síntesis de subbandas críticamente
decimadas con reconstrucción casi perfecta.

Rol en el pipeline RAVE:
    * El encoder observa la señal descompuesta en N_bands canales a Fs/N_bands.
    * El decoder produce N_bands muestras por paso de convolución. A 16 kHz
      con N_bands=16, cada subbanda corre a 1 kHz y las convoluciones del
      decoder cuestan 16x menos por segundo de audio.
    * La síntesis final es un banco lineal fijo, no entrenable.

Diseño:
    1. Prototipo pasa-bajos sinc * Kaiser, longitud L = taps + 1 (impar) con
       taps par para preservar simetría.
    2. Modulación cosenoidal de Nguyen: fase (-1)^k * pi/4 para análisis,
       fase opuesta para síntesis.
    3. Búsqueda en cuadrícula sobre (cutoff, beta) minimizando el error de
       reconstrucción sobre señales de banda ancha. Esta métrica captura
       correctamente la variabilidad periódica del banco (que la respuesta
       impulso sola oculta).

Estructura de conv:
    Análisis: filtrar con L filtros y luego diezmar con un kernel delta de
    stride n_bands. Separar el filtro del diezmado evita ambigüedades de
    correlación vs convolución.

    Síntesis: interpolar con un kernel delta escalado por n_bands (compensa
    la caída de energía por inserción de ceros) y combinar bandas con un
    conv1d cuyo peso tiene forma (1, n_bands, L).

Retardo total: taps muestras (grupo delay del prototipo simétrico), absorbido
por el padding simétrico taps//2 en cada extremo.

La calidad de reconstrucción tras optimización, con n_bands=16 y taps=254,
alcanza SNR alrededor de 30 dB y ganancia unitaria. Con taps menores la SNR
cae y aparece error de ganancia que el decoder tendría que aprender a
compensar.
'''

from __future__ import annotations

import numpy as np
import praxis.log as log
import torch
import torch.nn as nn
import torch.nn.functional as F


def _kaiser_beta(attenuation_db: float) -> float:
    '''Beta de la ventana Kaiser dada la atenuación de banda de detención.'''
    if attenuation_db > 50:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= 21:
        a = attenuation_db - 21
        return 0.5842 * a**0.4 + 0.07886 * a
    return 0.0


def _design_prototype(taps: int, cutoff_ratio: float, beta: float) -> np.ndarray:
    '''Prototipo pasa-bajos sinc ventaneado con Kaiser, longitud taps + 1.'''
    omega_c = np.pi * cutoff_ratio
    n = np.arange(taps + 1) - taps / 2
    with np.errstate(invalid='ignore', divide='ignore'):
        h = np.sin(omega_c * n) / (np.pi * n)
    h[taps // 2] = cutoff_ratio
    window = np.kaiser(taps + 1, beta)
    return (h * window).astype(np.float32)


def _build_filter_bank(prototype: np.ndarray, n_bands: int, taps: int) -> tuple[np.ndarray, np.ndarray]:
    '''Modulación cosenoidal en n_bands filtros de análisis y síntesis.'''
    k = np.arange(n_bands)
    n = np.arange(taps + 1) - taps / 2
    phase = (-1.0) ** k * np.pi / 4.0
    arg = (2 * k[:, None] + 1) * np.pi / (2 * n_bands) * n[None, :]
    analysis = 2.0 * prototype[None, :] * np.cos(arg + phase[:, None])
    synthesis = 2.0 * prototype[None, :] * np.cos(arg - phase[:, None])
    return analysis.astype(np.float32), synthesis.astype(np.float32)


def _make_torch_bank(prototype: np.ndarray, n_bands: int, taps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    analysis, synthesis = _build_filter_bank(prototype, n_bands, taps)
    analysis_t = torch.from_numpy(analysis).unsqueeze(1)
    synthesis_t = torch.from_numpy(synthesis).unsqueeze(0)
    updown = torch.zeros(n_bands, n_bands, n_bands)
    for k in range(n_bands):
        updown[k, k, 0] = 1.0
    return analysis_t, synthesis_t, updown


def _roundtrip(x: torch.Tensor, analysis: torch.Tensor, synthesis: torch.Tensor, updown: torch.Tensor, n_bands: int, taps: int) -> torch.Tensor:
    pad = taps // 2
    xa = F.pad(x, (pad, pad))
    xa = F.conv1d(xa, analysis)
    bands = F.conv1d(xa, updown, stride=n_bands)
    xu = F.conv_transpose1d(bands, updown * n_bands, stride=n_bands)
    xu = F.pad(xu, (pad, pad))
    return F.conv1d(xu, synthesis)


def _reconstruction_snr(prototype: np.ndarray, n_bands: int, taps: int, T: int = 4096, n_trials: int = 1) -> tuple[float, float]:
    '''SNR de reconstrucción en dB y ganancia proyectada, medidas sobre ruido
    blanco. La ganancia debe converger a 1 en un banco bien diseñado.'''
    analysis, synthesis, updown = _make_torch_bank(prototype, n_bands, taps)
    snrs, gains = [], []
    for seed in range(n_trials):
        torch.manual_seed(seed)
        x = torch.randn(1, 1, T)
        y = _roundtrip(x, analysis, synthesis, updown, n_bands, taps)
        xnp = x.squeeze().numpy()
        ynp = y.squeeze().numpy()
        gain = float(np.dot(ynp, xnp) / (np.dot(xnp, xnp) + 1e-30))
        err = ynp - gain * xnp
        sig_power = ((gain * xnp) ** 2).mean()
        err_power = (err**2).mean() + 1e-30
        snrs.append(10 * np.log10(sig_power / err_power))
        gains.append(gain)
    return float(np.mean(snrs)), float(np.mean(gains))


def _optimize_prototype(n_bands: int, taps: int, attenuation_db: float, verbose: bool = False) -> np.ndarray:
    '''Búsqueda 2D sobre (cutoff, beta) maximizando SNR real de
    reconstrucción.

    El punto de partida para cutoff es 1/(2*n_bands), con exploración de
    ±30%. Para beta se explora ±3 alrededor del valor derivado de la
    atenuación objetivo.
    '''
    base_cutoff = 1.0 / (2.0 * n_bands)
    base_beta = _kaiser_beta(attenuation_db)
    best_h = None
    best_snr = -np.inf
    cutoffs = np.linspace(base_cutoff * 0.7, base_cutoff * 1.3, 41)
    betas = np.linspace(max(0.0, base_beta - 3), base_beta + 3, 7)
    for cutoff in cutoffs:
        if cutoff <= 0 or cutoff >= 0.5:
            continue
        for beta in betas:
            h = _design_prototype(taps, float(cutoff), float(beta))
            snr, _ = _reconstruction_snr(h, n_bands, taps, T=2048, n_trials=1)
            if snr > best_snr:
                best_snr = snr
                best_h = h
                if verbose:
                    log.info(f'  cutoff={cutoff:.5f} beta={beta:.2f} ' f'SNR={snr:.2f} dB')
    return best_h


class PQMF(nn.Module):
    '''Banco PQMF fijo (no entrenable) para análisis y síntesis.

    Análisis:  (B, 1, T)              -> (B, n_bands, T // n_bands)
    Síntesis:  (B, n_bands, T // n_bands) -> (B, 1, T)

    La longitud T debe ser múltiplo de n_bands. El diseño se optimiza en el
    constructor por búsqueda en cuadrícula sobre (cutoff, beta). Para
    n_bands=16 y taps=254 se obtienen alrededor de 30 dB de SNR de
    reconstrucción con ganancia unitaria.
    '''

    def __init__(self, n_bands: int = 16, taps: int = 254, attenuation_db: float = 100.0, verbose: bool = False):
        super().__init__()
        if n_bands < 2:
            raise ValueError('n_bands debe ser >= 2')
        if taps % 2 != 0:
            raise ValueError('taps debe ser par (L = taps + 1 impar)')
        self.n_bands = n_bands
        self.taps = taps
        self.pad = taps // 2

        prototype = _optimize_prototype(n_bands, taps, attenuation_db, verbose=verbose)
        snr, gain = _reconstruction_snr(prototype, n_bands, taps, T=8192, n_trials=3)
        self.reconstruction_snr_db = float(snr)
        self.roundtrip_gain = float(gain)

        analysis, synthesis, updown = _make_torch_bank(prototype, n_bands, taps)
        self.register_buffer('analysis_kernel', analysis)
        self.register_buffer('synthesis_kernel', synthesis)
        self.register_buffer('updown_kernel', updown)

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(1) != 1:
            raise ValueError(f'se esperaba (B, 1, T), se recibió {tuple(x.shape)}')
        if x.size(-1) % self.n_bands != 0:
            raise ValueError(f'longitud {x.size(-1)} no es múltiplo de ' f'n_bands={self.n_bands}')
        x = F.pad(x, (self.pad, self.pad))
        x = F.conv1d(x, self.analysis_kernel)
        return F.conv1d(x, self.updown_kernel, stride=self.n_bands)

    def synthesize(self, bands: torch.Tensor) -> torch.Tensor:
        if bands.dim() != 3 or bands.size(1) != self.n_bands:
            raise ValueError(f'se esperaba (B, {self.n_bands}, T), se recibió ' f'{tuple(bands.shape)}')
        x = F.conv_transpose1d(bands, self.updown_kernel * self.n_bands, stride=self.n_bands)
        x = F.pad(x, (self.pad, self.pad))
        return F.conv1d(x, self.synthesis_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''Roundtrip para testing.'''
        return self.synthesize(self.analyze(x))
