'''Pérdidas para el VAE tipo RAVE, fases 1 y 2.

Fase 1 (VAE puro):
    * MultiScaleSTFTLoss: reconstrucción tiempo-frecuencia.
    * kl_free_bits: KL con floor por dimensión, evita colapso posterior.
    * BetaScheduler: warmup lineal.
    * RAVELoss: agregador de fase 1.

Fase 2 (adversarial):
    * hinge_loss_d: hinge para el discriminador.
    * hinge_loss_g: hinge para el generador.
    * feature_matching_loss: L1 sobre features intermedios del D real vs fake.
    * Estas funciones se combinan en engine.py con RAVELoss para producir la
      loss completa de fase 2.

Anti-colapso:
    * free_bits establece un piso por dimensión sobre la KL, evitando que el
      modelo apague dimensiones latentes durante el warmup del β. Sigue la
      formulación de Kingma+ 2016 (improved variational inference).
    * beta_max por defecto se baja a 1e-4 y warmup_steps sube a 100_000 para
      corpus pequeños. RAVE original ajusta β en escalas similares.
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
        x = x.reshape(-1, x.size(-1))
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
    '''KL(N(mu, sigma^2) || N(0, I)) forma cerrada, suma sobre D, promedio B, L.'''
    kl = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
    return kl.sum(dim=1).mean()


def kl_free_bits(mu: torch.Tensor, log_sigma: torch.Tensor, free_bits: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    '''KL con piso por dimensión (Kingma+ 2016).

    Args:
        mu, log_sigma: tensores (B, D, L).
        free_bits: piso en nats por dimensión. Si <= 0, se comporta como KL
            estándar (agregada como en kl_diagonal_gaussian).

    Returns:
        (kl_optimized, kl_raw):
            kl_optimized es la que entra al backward: max(kl_d, free_bits)
                sumado sobre D.
            kl_raw es la KL real sin piso, útil como diagnóstico.

    Nota:
        La KL por dimensión se promedia sobre batch y tiempo antes de aplicar
        el piso. Esto sigue la convención habitual en VAEs de audio y evita
        que dimensiones ocasionalmente ruidosas escapen del clamp.
    '''
    kl_per_element = 0.5 * (mu.pow(2) + torch.exp(2 * log_sigma) - 2 * log_sigma - 1)
    kl_per_dim = kl_per_element.mean(dim=(0, 2))
    kl_raw = kl_per_dim.sum()

    if free_bits > 0.0:
        kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
        kl_optimized = kl_clamped.sum()
    else:
        kl_optimized = kl_raw

    return kl_optimized, kl_raw.detach()


class BetaScheduler:
    '''Warmup lineal para el peso del KL.

    beta(step) = min(1, step / warmup_steps) * beta_max
    '''

    def __init__(self, beta_max: float = 1e-4, warmup_steps: int = 100_000):
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
        stft_weight: multiplicador del término STFT.
        waveform_weight: multiplicador del término L1 sobre la forma de onda.
        free_bits: piso en nats por dimensión para la KL.

    forward devuelve un dict con:
        total, stft_sc, stft_log_mag, waveform_l1, kl, kl_raw, beta
    '''

    def __init__(self, fft_sizes: tuple[int, ...] = (512, 1024, 2048), beta_max: float = 1e-4, warmup_steps: int = 100_000, stft_weight: float = 1.0, waveform_weight: float = 0.1, free_bits: float = 0.0):
        super().__init__()
        self.stft = MultiScaleSTFTLoss(fft_sizes)
        self.scheduler = BetaScheduler(beta_max, warmup_steps)
        self.stft_weight = float(stft_weight)
        self.waveform_weight = float(waveform_weight)
        self.free_bits = float(free_bits)

    def forward(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, log_sigma: torch.Tensor, step: int) -> dict[str, torch.Tensor]:
        stft_terms = self.stft(x, x_hat)
        stft_sc = stft_terms['stft_sc']
        stft_log_mag = stft_terms['stft_log_mag']

        waveform_l1 = F.l1_loss(x_hat, x)

        kl_opt, kl_raw = kl_free_bits(mu, log_sigma, self.free_bits)
        beta = self.scheduler(step)

        total = self.stft_weight * (stft_sc + stft_log_mag) + self.waveform_weight * waveform_l1 + beta * kl_opt

        return {
            'total': total,
            'stft_sc': stft_sc.detach(),
            'stft_log_mag': stft_log_mag.detach(),
            'waveform_l1': waveform_l1.detach(),
            'kl': kl_opt.detach(),
            'kl_raw': kl_raw,
            'beta': torch.tensor(beta, device=x.device),
        }


def hinge_loss_d(logits_real: list[torch.Tensor], logits_fake: list[torch.Tensor]) -> torch.Tensor:
    '''Hinge loss para el discriminador multi-escala.

    L_D = E[relu(1 - D(x))] + E[relu(1 + D(x_hat))]

    Args:
        logits_real: lista de logits por escala evaluados sobre audio real.
        logits_fake: lista de logits por escala evaluados sobre audio
            generado (debe pasarse detach para no propagar al generador).

    Returns:
        Escalar promedio sobre escalas.
    '''
    loss = logits_real[0].new_zeros(())
    for lr, lf in zip(logits_real, logits_fake):
        loss = loss + F.relu(1.0 - lr).mean() + F.relu(1.0 + lf).mean()
    return loss / len(logits_real)


def hinge_loss_g(logits_fake: list[torch.Tensor]) -> torch.Tensor:
    '''Hinge loss para el generador (no saturante).

    L_G_adv = -E[D(x_hat)]

    El generador quiere que D(x_hat) sea grande y positivo. Con hinge
    non-saturating, el gradiente no desaparece cuando D acierta.
    '''
    loss = logits_fake[0].new_zeros(())
    for lf in logits_fake:
        loss = loss - lf.mean()
    return loss / len(logits_fake)


def feature_matching_loss(features_real: list[list[torch.Tensor]], features_fake: list[list[torch.Tensor]]) -> torch.Tensor:
    '''L1 sobre feature maps intermedios del D, promediado por capa y escala.

    Args:
        features_real: features_real[i][j] es el j-ésimo feature de la
            i-ésima escala evaluado sobre audio real.
        features_fake: análogo, sobre audio generado.

    Returns:
        Escalar. Estabiliza el entrenamiento del generador dando gradientes
        densos incluso cuando la loss adversarial ya está satisfecha.
    '''
    if len(features_real) != len(features_fake):
        raise ValueError('features_real y features_fake deben tener el mismo número de escalas')

    total = features_real[0][0].new_zeros(())
    n_pairs = 0
    for feats_r, feats_f in zip(features_real, features_fake):
        if len(feats_r) != len(feats_f):
            raise ValueError('discordancia en número de capas dentro de una escala')
        for fr, ff in zip(feats_r, feats_f):
            total = total + F.l1_loss(ff, fr.detach())
            n_pairs += 1
    return total / max(n_pairs, 1)
