'''Verificación del PQMF final.

Se prueban configuraciones típicas y se reporta:
    * SNR de reconstrucción sobre ruido blanco.
    * SNR sobre un chirp lineal (contenido tonal barrido).
    * Ganancia empírica del roundtrip.
    * Consistencia de formas.
    * Retardo medido por correlación cruzada.
'''
import numpy as np
import torch

from pqmf import PQMF


def snr_and_gain(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    gain = float(np.dot(y, x) / (np.dot(x, x) + 1e-30))
    err = y - gain * x
    sig_power = ((gain * x) ** 2).mean()
    err_power = (err ** 2).mean() + 1e-30
    return 10 * np.log10(sig_power / err_power), gain


def measure_delay(x: np.ndarray, y: np.ndarray) -> int:
    corr = np.correlate(y, x, mode='full')
    return int(np.argmax(np.abs(corr)) - (len(x) - 1))


def main() -> None:
    print(f'{"n":>3} {"taps":>4} {"SNR_r":>7} {"SNR_c":>7} '
          f'{"gain":>6} {"delay":>6}')
    print('-' * 42)
    for n_bands, taps in [(8, 62), (8, 126), (16, 126), (16, 254)]:
        pqmf = PQMF(n_bands=n_bands, taps=taps, attenuation_db=100.0)
        T = 8192

        noise = torch.randn(1, 1, T)
        y = pqmf(noise).squeeze().numpy()
        snr_n, gain = snr_and_gain(noise.squeeze().numpy(), y)

        t = torch.linspace(0, 1, T)
        chirp = torch.sin(2 * np.pi * (100 * t + 3000 * t ** 2))
        chirp = chirp.unsqueeze(0).unsqueeze(0).float()
        yc = pqmf(chirp).squeeze().numpy()
        snr_c, _ = snr_and_gain(chirp.squeeze().numpy(), yc)

        delay = measure_delay(noise.squeeze().numpy(), y)
        print(f'{n_bands:>3} {taps:>4} {snr_n:>7.2f} {snr_c:>7.2f} '
              f'{gain:>6.3f} {delay:>6}')

    print()
    print('Consistencia de formas (n_bands=16, taps=254):')
    pqmf = PQMF(n_bands=16, taps=254)
    for T in (1024, 4096, 16384):
        x = torch.randn(2, 1, T)
        bands = pqmf.analyze(x)
        y = pqmf.synthesize(bands)
        assert bands.shape == (2, 16, T // 16)
        assert y.shape == x.shape
        print(f'  T={T}: bands {tuple(bands.shape)}, y {tuple(y.shape)}')

    print()
    print(f'SNR de PQMF(16, 254) reportado por el módulo: '
          f'{pqmf.reconstruction_snr_db:.2f} dB, gain={pqmf.roundtrip_gain:.4f}')


if __name__ == '__main__':
    main()
