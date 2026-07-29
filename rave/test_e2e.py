'''Smoke tests end-to-end.

Verifican que:
    * Un ciclo corto de entrenamiento en fase 1 termina sin explotar.
    * Un ciclo corto de entrenamiento en fase 2 termina sin explotar.
    * Los checkpoints se guardan y se pueden cargar.
    * La reanudación fase 1 -> fase 2 en modo finetune funciona.

No verifica calidad: 5 pasos no son suficientes para converger. Solo
verifica que las piezas encajan.
'''

from __future__ import annotations
from .engine import TrainConfig, train

import numpy as np
import pathlib
import soundfile as sf
import tempfile
import torch


def _write_dummy_audio(path: pathlib.Path, duration_s: float = 30.0, sr: int = 16_000):
    '''Genera audio sintético: mezcla de senos + ruido, escrito a disco.'''
    t = np.arange(int(duration_s * sr)) / sr
    y = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.randn(len(t))
    y = y.astype(np.float32)
    sf.write(path, y, sr)


def _base_config(out_dir: pathlib.Path, audio_path: pathlib.Path) -> TrainConfig:
    return TrainConfig(
        audio_paths=[str(audio_path)],
        out_dir=out_dir,
        segment_length=8192,
        hop_length=4096,
        batch_size=2,
        num_workers=0,
        hidden_channels=32,
        n_res_per_block=2,
        latent_dim=32,
        beta_max=1e-4,
        warmup_steps=100,
        free_bits=0.05,
        n_steps=5,
        log_every=1,
        val_every=5,
        ckpt_every=5,
        keep_last_ckpts=1,
        device='cpu',
    )


def test_phase1_smoke():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        audio = tmp / 'in.wav'
        _write_dummy_audio(audio)

        config = _base_config(tmp / 'run', audio)
        config.phase = 1
        final = train(config)

        assert final.exists(), f'checkpoint fase 1 no se guardó: {final}'
        ckpt = torch.load(final, map_location='cpu', weights_only=False)
        assert 'model' in ckpt
        assert ckpt.get('step') == 5
        assert (tmp / 'run' / 'train_log.csv').exists()
        assert (tmp / 'run' / 'val_log.csv').exists()


def test_phase2_smoke_from_scratch():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        audio = tmp / 'in.wav'
        _write_dummy_audio(audio)

        config = _base_config(tmp / 'run', audio)
        config.phase = 2
        final = train(config)

        assert final.exists()
        ckpt = torch.load(final, map_location='cpu', weights_only=False)
        assert 'model' in ckpt
        assert 'discriminator' in ckpt
        assert 'optimizer_d' in ckpt


def test_phase1_to_phase2_finetune():
    '''Entrena fase 1, luego reanuda en fase 2 con modo finetune.'''
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        audio = tmp / 'in.wav'
        _write_dummy_audio(audio)

        cfg_p1 = _base_config(tmp / 'run_p1', audio)
        cfg_p1.phase = 1
        p1_ckpt = train(cfg_p1)

        cfg_p2 = _base_config(tmp / 'run_p2', audio)
        cfg_p2.phase = 2
        cfg_p2.resume_from = p1_ckpt
        cfg_p2.resume_mode = 'finetune'
        p2_ckpt = train(cfg_p2)

        assert p2_ckpt.exists()
        ckpt = torch.load(p2_ckpt, map_location='cpu', weights_only=False)
        # con finetune el step arranca en 0, así que después de 5 pasos: 5
        assert ckpt.get('step') == 5


def test_reconstruct_after_phase1():
    '''Carga un checkpoint de fase 1 y reconstruye un archivo.'''
    from .generate import load_model, reconstruct

    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        audio_in = tmp / 'in.wav'
        _write_dummy_audio(audio_in, duration_s=5.0)

        config = _base_config(tmp / 'run', audio_in)
        config.phase = 1
        final = train(config)

        model, cfg = load_model(final, device=torch.device('cpu'))
        audio_out = tmp / 'out.wav'
        reconstruct(model, audio_in, audio_out, sample_rate=cfg['sample_rate'], device=torch.device('cpu'))
        assert audio_out.exists()
        y, sr = sf.read(audio_out)
        assert sr == 16_000
        assert len(y) > 0
