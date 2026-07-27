'''Pérdidas para el VAE tipo RAVE.

Componentes:
    * MultiScaleSTFTLoss: reconstrucción en el dominio tiempo-frecuencia,
      con spectral convergence y log-magnitud L1 sobre tres escalas FFT.
    * kl_diagonal_gaussian: KL forma cerrada de N(mu, sigma^2) contra N(0, I).
    * BetaScheduler: warmup lineal del peso del KL.
    * RAVELoss: agregador. Devuelve un dict con todos los componentes para
      logging separado.

Pesos por defecto:
    reconstrucción STFT (spectral convergence + log mag) : 1.0
    reconstrucción dominio tiempo (L1)                   : 0.1
    KL warmed up hasta β_max = 0.1 en 10_000 pasos

Estos valores siguen la convención del paper RAVE. Ajustar en tiempo de
construcción según el balance observado en las curvas de entrenamiento.
'''

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleSTFTLoss(nn.Module):
    '''Pérdida STFT multi-escala.

    Para cada escala FFT calcula:
        L_sc  = || |S| - |S_hat| ||_F / || |S| ||_F     (spectral convergence)
        L_mag = || log(|S| + eps) - log(|S_hat| + eps) ||_1  (log magnitud)

    La suma sobre escalas se normaliza por el número de escalas.
    '''

    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1024, 2048), hop_ratio: float = 0.25, log_epsilon: float = 1e-7):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.hop_sizes = tuple(int(n * hop_ratio) for n in fft_sizes)
        self.log_epsilon = log_epsilon
        for i, n_fft in enumerate(fft_sizes):
            self.register_buffer(f'window_{i}', torch.hann_window(n_fft), persistent=False)

    def _stft_magnitude(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        n_fft = self.fft_sizes[idx]
        hop = self.hop_sizes[idx]
        window = getattr(self, f'window_{idx}')
        x = x.reshape(-1, x.size(-1))  # (B*C, T)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True, center=True, pad_mode='reflect')
        return spec.abs()

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor) -> dict[str, torch.Tensor]:
        sc_total = x.new_zeros(())
        mag_total = x.new_zeros(())
        for i in range(len(self.fft_sizes)):
            S = self._stft_magnitude(x, i)
            S_hat = self._stft_magnitude(x_hat, i)
            num = torch.linalg.norm(S - S_hat, dim=(-2, -1))
            den = torch.linalg.norm(S, dim=(-2, -1)) + 1e-7
            sc_total = sc_total + (num / den).mean()
            mag_total = mag_total + F.l1_loss(torch.log(S_hat + self.log_epsilon), torch.log(S + self.log_epsilon))
        n = len(self.fft_sizes)
        return {'stft_sc': sc_total / n, 'stft_log_mag': mag_total / n}


def kl_diagonal_gaussian(mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
    '''KL(N(mu, sigma^2) || N(0, I)) forma cerrada.

    Para gaussianas diagonales:
        KL = 0.5 * sum_d (mu_d^2 + sigma_d^2 - 1 - 2 log sigma_d)

    Args:
        mu, log_sigma: tensores de forma (B, D, L).

    Returns:
        Escalar: suma sobre D, promedio sobre B y L.
    '''
    kl = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
    return kl.sum(dim=1).mean()


class BetaScheduler:
    '''Warmup lineal para el peso del KL.

    beta(step) = min(1, step / warmup_steps) * beta_max

    No es un nn.Module: se llama con el step actual y se pasa el resultado
    como peso escalar en el forward de la pérdida agregada.
    '''

    def __init__(self, beta_max: float = 0.1, warmup_steps: int = 10_000):
        if warmup_steps < 1:
            raise ValueError('warmup_steps debe ser >= 1')
        self.beta_max = float(beta_max)
        self.warmup_steps = int(warmup_steps)

    def __call__(self, step: int) -> float:
        if step >= self.warmup_steps:
            return self.beta_max
        return self.beta_max * step / self.warmup_steps


class RAVELoss(nn.Module):
    '''Agregador de pérdidas del VAE.

    Args:
        fft_sizes: tamaños FFT para MultiScaleSTFTLoss.
        beta_max: peso final del término KL después del warmup.
        warmup_steps: pasos del warmup lineal para beta.
        stft_weight: multiplicador del término STFT (spectral convergence +
            log mag).
        waveform_weight: multiplicador del término L1 sobre la forma de onda.

    forward devuelve un dict con:
        total, stft_sc, stft_log_mag, waveform_l1, kl, beta
    '''

    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1024, 2048), beta_max: float = 0.1, warmup_steps: int = 10_000, stft_weight: float = 1.0, waveform_weight: float = 0.1):
        super().__init__()
        self.stft = MultiScaleSTFTLoss(fft_sizes)
        self.scheduler = BetaScheduler(beta_max, warmup_steps)
        self.stft_weight = float(stft_weight)
        self.waveform_weight = float(waveform_weight)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        stft_terms = self.stft(x, x_hat)
        stft_sc = stft_terms['stft_sc']
        stft_log_mag = stft_terms['stft_log_mag']

        waveform_l1 = F.l1_loss(x_hat, x)

        kl = kl_diagonal_gaussian(mu, log_sigma)
        beta = self.scheduler(step)

        total = self.stft_weight * (stft_sc + stft_log_mag) + self.waveform_weight * waveform_l1 + beta * kl

        return {
            'total': total,
            'stft_sc': stft_sc.detach(),
            'stft_log_mag': stft_log_mag.detach(),
            'waveform_l1': waveform_l1.detach(),
            'kl': kl.detach(),
            'beta': torch.tensor(beta, device=x.device),
        }
