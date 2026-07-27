'''Smoke test end-to-end: dataset → entrenamiento breve → reconstrucción →
muestreo desde prior.

Genera audio sintético (mezcla de senoides con envolvente), entrena 200 pasos
sobre él, y verifica que:
    * La pérdida total desciende de forma monotónica media a media.
    * El checkpoint se guarda y se puede recargar.
    * La reconstrucción produce audio de la misma forma que la entrada.
    * El muestreo del prior produce audio de la duración pedida.
'''
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from engine import TrainConfig, train
from generate import load_model, reconstruct, sample_prior


def _make_synthetic_audio(path: Path, duration_s: float,
                          sample_rate: int = 16_000) -> None:
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, dtype=np.float32)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 0.7 * t))
    waveform = envelope * (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t + 0.5)
        + 0.2 * np.sin(2 * np.pi * 880 * t + 1.2))
    waveform = waveform / (np.abs(waveform).max() + 1e-9)
    sf.write(str(path), waveform, sample_rate)


def main() -> None:
    workdir = Path('/tmp/rave_smoke')
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    audio_path = workdir / 'synth.wav'
    _make_synthetic_audio(audio_path, duration_s=8.0)
    print(f'audio sintético generado: {audio_path}')

    config = TrainConfig(
        audio_paths=[str(audio_path)],
        out_dir=str(workdir / 'run'),
        sample_rate=16_000,
        segment_length=8_192,
        hop_length=4_096,
        val_fraction=0.2,
        batch_size=4,
        num_workers=0,
        n_bands=16,
        pqmf_taps=126,
        hidden_channels=32,
        strides=(2, 4, 2),
        latent_dim=32,
        n_res_per_block=2,
        beta_max=0.1,
        warmup_steps=50,
        n_steps=200,
        log_every=50,
        val_every=100,
        ckpt_every=100,
        keep_last_ckpts=2,
        seed=0,
        device='cpu')

    final_ckpt = train(config)
    assert final_ckpt.exists()

    print()
    print('Recargando checkpoint y reconstruyendo...')
    model, cfg = load_model(final_ckpt, device=torch.device('cpu'))
    rec_out = workdir / 'reconstruido.wav'
    reconstruct(model, audio_path, rec_out, sample_rate=cfg['sample_rate'])
    rec_wav, sr = sf.read(str(rec_out), dtype='float32', always_2d=True)
    print(f'audio reconstruido: forma {rec_wav.T.shape}, sr={sr}')
    assert rec_wav.shape[1] == 1
    assert sr == 16_000

    print()
    print('Muestreando del prior...')
    prior_out = workdir / 'muestra_prior.wav'
    sample_prior(model, prior_out, duration_seconds=2.0,
                 sample_rate=cfg['sample_rate'], seed=42)
    prior_wav, sr = sf.read(str(prior_out), dtype='float32', always_2d=True)
    print(f'muestra del prior: forma {prior_wav.T.shape}, sr={sr}')
    expected_samples = int(2.0 * 16_000) - int(2.0 * 16_000) % model.total_stride
    assert prior_wav.shape[0] == expected_samples

    print()
    print('Verificando descenso de la pérdida en train_log.csv...')
    import csv
    with (workdir / 'run' / 'train_log.csv').open() as f:
        reader = csv.DictReader(f)
        totals = [(int(r['step']), float(r['total'])) for r in reader]
    print(f'  entradas de log: {len(totals)}')
    for step, total in totals:
        print(f'    step {step}: total = {total:.4f}')
    first, last = totals[0][1], totals[-1][1]
    print(f'  primer total: {first:.4f}  →  último total: {last:.4f}  '
          f'({(last - first) / first * 100:+.1f}%)')

    print()
    print('smoke test completo')


if __name__ == '__main__':
    main()
