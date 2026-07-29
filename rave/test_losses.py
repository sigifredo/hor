'''Verificación de losses de fase 1 y fase 2.

Comprueba:
    * KL forma cerrada en casos límite.
    * Free bits: piso por dimensión respetado.
    * BetaScheduler produce warmup lineal correcto.
    * MultiScaleSTFTLoss: cero en igualdad, positivo en desigualdad.
    * Hinge loss del D: positiva y crece cuando D falla.
    * Hinge loss del G: signo esperado.
    * Feature matching: cero en features idénticos.
    * Gradientes fluyen end-to-end.
'''

from __future__ import annotations
from .discriminator import MultiScaleDiscriminator
from .losses import (
    BetaScheduler,
    MultiScaleSTFTLoss,
    RAVELoss,
    feature_matching_loss,
    hinge_loss_d,
    hinge_loss_g,
    kl_diagonal_gaussian,
    kl_free_bits,
)
from .model import RAVE

import torch


def test_kl_prior_case() -> None:
    B, D, L = 4, 64, 32
    mu = torch.zeros(B, D, L)
    log_sigma = torch.zeros(B, D, L)
    assert abs(kl_diagonal_gaussian(mu, log_sigma).item()) < 1e-6

    mu = torch.full((B, D, L), 2.0)
    log_sigma = torch.zeros(B, D, L)
    assert abs(kl_diagonal_gaussian(mu, log_sigma).item() - 128) < 1e-3

    mu = torch.zeros(B, D, L)
    log_sigma = torch.full((B, D, L), -1.0)
    expected = 0.5 * 64 * (torch.exp(torch.tensor(-2.0)).item() + 1)
    assert abs(kl_diagonal_gaussian(mu, log_sigma).item() - expected) < 1e-3


def test_kl_free_bits_floor():
    B, D, L = 4, 16, 32
    mu = torch.zeros(B, D, L)
    log_sigma = torch.zeros(B, D, L)

    # sin free bits: KL = 0
    kl_opt, kl_raw = kl_free_bits(mu, log_sigma, free_bits=0.0)
    assert abs(kl_opt.item()) < 1e-6
    assert abs(kl_raw.item()) < 1e-6

    # con free_bits=0.1 y KL/dim=0: piso => KL_opt = 0.1 * D = 1.6
    kl_opt, kl_raw = kl_free_bits(mu, log_sigma, free_bits=0.1)
    assert abs(kl_opt.item() - 0.1 * D) < 1e-5
    assert abs(kl_raw.item()) < 1e-6

    # con KL/dim alto, free_bits no debe subir el valor real
    mu = torch.full((B, D, L), 1.0)
    kl_opt, kl_raw = kl_free_bits(mu, log_sigma, free_bits=0.1)
    # cada dim aporta 0.5 * 1^2 = 0.5 >> 0.1
    assert abs(kl_opt.item() - kl_raw.item()) < 1e-5


def test_kl_free_bits_returns_detached_raw():
    mu = torch.zeros(2, 4, 8, requires_grad=True)
    log_sigma = torch.zeros(2, 4, 8, requires_grad=True)
    kl_opt, kl_raw = kl_free_bits(mu, log_sigma, free_bits=0.1)
    assert kl_opt.requires_grad
    assert not kl_raw.requires_grad


def test_beta_scheduler() -> None:
    sched = BetaScheduler(beta_max=0.1, warmup_steps=1000)
    for step, expected in [(0, 0.0), (500, 0.05), (999, 0.0999), (1000, 0.1), (10_000, 0.1)]:
        assert abs(sched(step) - expected) < 1e-4


def test_stft_zero_on_equal_input() -> None:
    stft = MultiScaleSTFTLoss()
    x = torch.randn(2, 1, 16384)
    terms = stft(x, x.clone())
    assert terms['stft_sc'].item() < 1e-5
    assert terms['stft_log_mag'].item() < 1e-5


def test_stft_positive_on_noise() -> None:
    stft = MultiScaleSTFTLoss()
    x = torch.randn(2, 1, 16384)
    x_hat = torch.randn(2, 1, 16384)
    terms = stft(x, x_hat)
    assert terms['stft_sc'].item() > 0.5
    assert terms['stft_log_mag'].item() > 0.1


def test_hinge_d_zero_when_perfect():
    '''Cuando D acierta con margen (real >> 1, fake << -1), la hinge es 0.'''
    logits_real = [torch.full((2, 1, 8), 5.0)]
    logits_fake = [torch.full((2, 1, 8), -5.0)]
    loss = hinge_loss_d(logits_real, logits_fake)
    assert abs(loss.item()) < 1e-6


def test_hinge_d_positive_when_wrong():
    logits_real = [torch.full((2, 1, 8), -1.0)]
    logits_fake = [torch.full((2, 1, 8), 1.0)]
    loss = hinge_loss_d(logits_real, logits_fake)
    # relu(1 - (-1)) + relu(1 + 1) = 2 + 2 = 4, promedio sobre 1 escala = 4
    assert abs(loss.item() - 4.0) < 1e-6


def test_hinge_g_wants_high_fake_score():
    '''L_G = -mean(logits_fake). Con logits altos, loss muy negativa.'''
    logits_fake = [torch.full((2, 1, 8), 5.0)]
    loss = hinge_loss_g(logits_fake)
    assert abs(loss.item() - (-5.0)) < 1e-6


def test_feature_matching_zero_on_identical():
    features_real = [[torch.randn(2, 4, 16) for _ in range(3)] for _ in range(2)]
    features_fake = [[f.clone() for f in scale] for scale in features_real]
    loss = feature_matching_loss(features_real, features_fake)
    assert abs(loss.item()) < 1e-6


def test_feature_matching_positive_on_difference():
    features_real = [[torch.zeros(2, 4, 16) for _ in range(3)] for _ in range(2)]
    features_fake = [[torch.ones(2, 4, 16) for _ in range(3)] for _ in range(2)]
    loss = feature_matching_loss(features_real, features_fake)
    assert abs(loss.item() - 1.0) < 1e-5


def test_end_to_end_gradients_phase1() -> None:
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=32, n_res_per_block=2)
    loss_fn = RAVELoss(fft_sizes=(512, 1024, 2048), beta_max=0.1, warmup_steps=1000, free_bits=0.1)
    x = torch.randn(2, 1, 4096) * 0.3
    x_hat, mu, log_sigma = model(x)
    losses = loss_fn(x, x_hat, mu, log_sigma, step=5000)
    losses['total'].backward()
    n_none = sum(1 for _, p in model.named_parameters() if p.grad is None)
    assert n_none == 0


def test_end_to_end_gradients_phase2_g() -> None:
    '''Loss del generador con adversarial y FM: gradientes fluyen a G.'''
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=32, n_res_per_block=2)
    discriminator = MultiScaleDiscriminator()
    loss_fn = RAVELoss(fft_sizes=(512, 1024, 2048), beta_max=1e-4, warmup_steps=1000, free_bits=0.1)

    x = torch.randn(2, 1, 4096) * 0.3
    x_hat, mu, log_sigma = model(x)
    rec = loss_fn(x, x_hat, mu, log_sigma, step=100)

    d_out_real = discriminator(x)
    d_out_fake = discriminator(x_hat)
    features_real = [f for f, _ in d_out_real]
    features_fake = [f for f, _ in d_out_fake]
    logits_fake = [l for _, l in d_out_fake]

    g_adv = hinge_loss_g(logits_fake)
    fm = feature_matching_loss(features_real, features_fake)
    total_g = rec['total'] + 1.0 * g_adv + 10.0 * fm
    total_g.backward()

    n_none_g = sum(1 for _, p in model.named_parameters() if p.grad is None)
    assert n_none_g == 0


def test_beta_zero_at_step_0() -> None:
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=32, n_res_per_block=2)
    loss_fn = RAVELoss(fft_sizes=(512, 1024, 2048), beta_max=0.1, warmup_steps=1000, free_bits=0.0)
    x = torch.randn(2, 1, 4096) * 0.3
    x_hat, mu, log_sigma = model(x)
    losses = loss_fn(x, x_hat, mu, log_sigma, step=0)
    assert losses['beta'].item() == 0.0
    expected = losses['stft_sc'].item() + losses['stft_log_mag'].item() + 0.1 * losses['waveform_l1'].item()
    assert abs(losses['total'].item() - expected) < 1e-4
