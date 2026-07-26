'''Verificación de losses.

Comprueba:
    * Cada término es finito y no negativo cuando corresponde.
    * KL con mu=0, log_sigma=0 (posterior = prior) es exactamente 0.
    * KL crece cuando la posterior se aleja del prior.
    * BetaScheduler produce warmup lineal correcto.
    * Gradientes fluyen desde total hasta el modelo.
    * total es diferenciable end-to-end.
'''
import torch

from losses import (BetaScheduler, MultiScaleSTFTLoss, RAVELoss,
                    kl_diagonal_gaussian)
from model import RAVE


def test_kl_prior_case() -> None:
    print('=== KL: casos límite ===')
    B, D, L = 4, 64, 32
    # Posterior = prior: mu=0, log_sigma=0 → sigma=1
    mu = torch.zeros(B, D, L)
    log_sigma = torch.zeros(B, D, L)
    kl = kl_diagonal_gaussian(mu, log_sigma)
    print(f'  mu=0, log_sigma=0: KL = {kl.item():.6e}  (esperado 0)')
    assert abs(kl.item()) < 1e-6

    # Posterior alejada: mu=2, log_sigma=0
    mu = torch.full((B, D, L), 2.0)
    log_sigma = torch.zeros(B, D, L)
    kl = kl_diagonal_gaussian(mu, log_sigma)
    # KL analítica = 0.5 * D * (mu^2) = 0.5 * 64 * 4 = 128
    print(f'  mu=2, log_sigma=0: KL = {kl.item():.4f}  (esperado 128)')
    assert abs(kl.item() - 128) < 1e-3

    # Posterior con sigma pequeño: mu=0, log_sigma=-1
    mu = torch.zeros(B, D, L)
    log_sigma = torch.full((B, D, L), -1.0)
    kl = kl_diagonal_gaussian(mu, log_sigma)
    # KL analítica = 0.5 * D * (exp(-2) - 2*(-1) - 1) = 0.5 * 64 * (0.1353 + 1)
    expected = 0.5 * 64 * (torch.exp(torch.tensor(-2.0)).item() + 1)
    print(f'  mu=0, log_sigma=-1: KL = {kl.item():.4f}  (esperado '
          f'{expected:.4f})')
    assert abs(kl.item() - expected) < 1e-3


def test_beta_scheduler() -> None:
    print('\n=== BetaScheduler ===')
    sched = BetaScheduler(beta_max=0.1, warmup_steps=1000)
    checkpoints = [(0, 0.0), (500, 0.05), (999, 0.0999), (1000, 0.1),
                   (10_000, 0.1)]
    for step, expected in checkpoints:
        got = sched(step)
        print(f'  step={step:>6}: beta = {got:.4f}  (esperado {expected:.4f})')
        assert abs(got - expected) < 1e-4


def test_stft_shapes_and_zero() -> None:
    print('\n=== MultiScaleSTFTLoss ===')
    stft = MultiScaleSTFTLoss()
    T = 16384
    x = torch.randn(2, 1, T)
    # Con x_hat = x, las dos pérdidas STFT deben ser cero (o casi)
    terms = stft(x, x.clone())
    print(f'  x_hat = x: sc = {terms["stft_sc"].item():.4e}, '
          f'log_mag = {terms["stft_log_mag"].item():.4e}')
    assert terms['stft_sc'].item() < 1e-5
    assert terms['stft_log_mag'].item() < 1e-5

    # Con x_hat = ruido, ambas pérdidas positivas y significativas
    x_hat = torch.randn(2, 1, T)
    terms = stft(x, x_hat)
    print(f'  x_hat = ruido: sc = {terms["stft_sc"].item():.4f}, '
          f'log_mag = {terms["stft_log_mag"].item():.4f}')
    assert terms['stft_sc'].item() > 0.5
    assert terms['stft_log_mag'].item() > 0.1


def test_end_to_end_gradients() -> None:
    print('\n=== Gradientes end-to-end ===')
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126,
                 hidden_channels=32, strides=(2, 4, 2),
                 latent_dim=32, n_res_per_block=2)
    loss_fn = RAVELoss(fft_sizes=(512, 1024, 2048),
                       beta_max=0.1, warmup_steps=1000)
    x = torch.randn(2, 1, 4096) * 0.3
    x_hat, mu, log_sigma = model(x)
    losses = loss_fn(x, x_hat, mu, log_sigma, step=5000)
    print(f'  step=5000, beta = {losses["beta"].item():.4f}')
    print(f'  total          = {losses["total"].item():.4f}')
    print(f'  stft_sc        = {losses["stft_sc"].item():.4f}')
    print(f'  stft_log_mag   = {losses["stft_log_mag"].item():.4f}')
    print(f'  waveform_l1    = {losses["waveform_l1"].item():.4f}')
    print(f'  kl             = {losses["kl"].item():.4f}')

    losses['total'].backward()
    n_none = sum(1 for _, p in model.named_parameters() if p.grad is None)
    n_zero = sum(1 for _, p in model.named_parameters()
                 if p.grad is not None and p.grad.abs().max().item() == 0)
    print(f'  parámetros grad=None: {n_none}')
    print(f'  parámetros grad=0   : {n_zero}')
    assert n_none == 0


def test_beta_zero_at_step_0() -> None:
    print('\n=== En step=0, KL no contribuye al total ===')
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126,
                 hidden_channels=32, strides=(2, 4, 2),
                 latent_dim=32, n_res_per_block=2)
    loss_fn = RAVELoss(fft_sizes=(512, 1024, 2048),
                       beta_max=0.1, warmup_steps=1000)
    x = torch.randn(2, 1, 4096) * 0.3
    x_hat, mu, log_sigma = model(x)
    losses = loss_fn(x, x_hat, mu, log_sigma, step=0)
    print(f'  beta = {losses["beta"].item():.4e}')
    assert losses['beta'].item() == 0.0
    # total = stft_weight * (sc + mag) + waveform_weight * l1
    expected = (losses['stft_sc'].item() + losses['stft_log_mag'].item()
                + 0.1 * losses['waveform_l1'].item())
    print(f'  total = {losses["total"].item():.4f} (esperado {expected:.4f})')
    assert abs(losses['total'].item() - expected) < 1e-4


if __name__ == '__main__':
    test_kl_prior_case()
    test_beta_scheduler()
    test_stft_shapes_and_zero()
    test_end_to_end_gradients()
    test_beta_zero_at_step_0()
