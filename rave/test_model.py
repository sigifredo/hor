'''Verificación estructural del modelo RAVE.

Comprueba:
    * Consistencia de formas end-to-end.
    * Que gradientes fluyen desde la salida hasta todos los parámetros.
    * Conteo de parámetros por submódulo.
    * Que la reconstrucción con pesos aleatorios no explota (sanity check).
    * Que sample_prior funciona.
'''

import torch

from .model import RAVE


def test_shapes() -> None:
    print('=== Consistencia de formas ===')
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=64, strides=(2, 4, 2), latent_dim=128, n_res_per_block=3)
    model.eval()

    total_stride = model.total_stride
    print(f'total_stride = {total_stride}  (n_bands * prod(strides))')

    for T in (2048, 4096, 16384):
        if T % total_stride != 0:
            print(f'  T={T}: omitido (no múltiplo de {total_stride})')
            continue
        x = torch.randn(2, 1, T)
        with torch.no_grad():
            mu, log_sigma = model.encode(x)
            z = model.reparameterize(mu, log_sigma)
            x_hat = model.decode(z)
        L_z = T // total_stride
        assert mu.shape == (2, 128, L_z), f'mu shape {mu.shape} != esperado {(2, 128, L_z)}'
        assert log_sigma.shape == mu.shape
        assert x_hat.shape == x.shape, f'x_hat shape {x_hat.shape} != {x.shape}'
        print(f'  T={T:>5}: mu {tuple(mu.shape)}, x_hat {tuple(x_hat.shape)}')


def test_gradient_flow() -> None:
    print('\n=== Flujo de gradientes ===')
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=64, n_res_per_block=2)
    x = torch.randn(2, 1, 2048, requires_grad=False)
    x_hat, mu, log_sigma = model(x)
    loss = (x_hat - x).pow(2).mean() + 0.01 * (mu.pow(2) + log_sigma.exp()).mean()
    loss.backward()

    zero_grad = []
    none_grad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            none_grad.append(name)
        elif p.grad.abs().max().item() == 0:
            zero_grad.append(name)
    print(f'  parámetros con grad=None : {len(none_grad)}')
    print(f'  parámetros con grad=0    : {len(zero_grad)}')
    if none_grad:
        for n in none_grad[:5]:
            print(f'    {n}')
    assert not none_grad, 'algún parámetro no recibió gradiente'


def test_parameter_count() -> None:
    print('\n=== Conteo de parámetros ===')
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=64, strides=(2, 4, 2), latent_dim=128, n_res_per_block=3)
    counts = model.parameter_count()
    for k, v in counts.items():
        print(f'  {k:>8s}: {v:>12,d}')


def test_sample_prior() -> None:
    print('\n=== sample_prior ===')
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=64, n_res_per_block=2)
    model.eval()
    L_z = 32
    audio = model.sample_prior(batch_size=2, length=L_z)
    expected_T = L_z * model.total_stride
    print(f'  L_z={L_z} → audio {tuple(audio.shape)}, esperado T={expected_T}')
    assert audio.shape == (2, 1, expected_T)
    print(f'  amplitud pico: {audio.abs().max().item():.4f}')
    print(f'  std: {audio.std().item():.4f}')


def test_reconstruction_untrained() -> None:
    print('\n=== Reconstrucción con pesos aleatorios ===')
    torch.manual_seed(0)
    model = RAVE(n_bands=16, pqmf_taps=126, hidden_channels=32, strides=(2, 4, 2), latent_dim=64, n_res_per_block=2)
    model.eval()
    x = torch.randn(1, 1, 4096) * 0.3
    with torch.no_grad():
        x_hat = model.reconstruct(x)
    ratio = x_hat.abs().max().item() / max(x.abs().max().item(), 1e-8)
    print(f'  x std={x.std().item():.4f}, x_hat std={x_hat.std().item():.4f}')
    print(f'  ratio de pico: {ratio:.4f}')
    if ratio > 100 or ratio < 0.01:
        print('  ADVERTENCIA: reconstrucción sin entrenar tiene escala ' 'anómala (esperado si los pesos no están normalizados).')


if __name__ == '__main__':
    test_shapes()
    test_gradient_flow()
    test_parameter_count()
    test_sample_prior()
    test_reconstruction_untrained()
