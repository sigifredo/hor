'''Inferencia del VAE tipo RAVE.

Dos modos:
    * reconstruct: audio → encode → mu → decode → audio.
      Usa mu (no reparametriza) para reconstrucción determinista.

    * sample_prior: z ~ N(0, I) → decode → audio.
      Con corpus pequeño, esta ruta produce artefactos consistentes con la
      variedad del latente entrenada.

Convenciones:
    * El modelo requiere que la longitud del audio sea múltiplo de
      total_stride. Los archivos se recortan al múltiplo más cercano.
    * La generación se hace en bloques del tamaño máximo que la VRAM permita,
      pero por simplicidad el default carga todo el archivo en memoria.
'''

from __future__ import annotations

from .model import RAVE
from pathlib import Path

import praxis.log as log
import soundfile as sf
import torch
import torch.nn.functional as F


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> tuple[RAVE, dict]:
    '''Carga un checkpoint y reconstruye el modelo con su configuración.'''
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ckpt['config']
    model = RAVE(n_bands=config['n_bands'], pqmf_taps=config['pqmf_taps'], hidden_channels=config['hidden_channels'], strides=tuple(config['strides']), latent_dim=config['latent_dim'], n_res_per_block=config['n_res_per_block']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, config


def _load_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).contiguous()
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        n_out = int(waveform.size(-1) * target_sr / sr)
        waveform = F.interpolate(waveform.unsqueeze(0), size=n_out, mode='linear', align_corners=False).squeeze(0)
    peak = waveform.abs().max().clamp(min=1e-6)
    return waveform / peak


def _save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)
    sf.write(str(path), waveform.cpu().numpy(), sample_rate)


@torch.no_grad()
def reconstruct(model: RAVE, input_path: str | Path, output_path: str | Path, sample_rate: int = 16_000, device: torch.device | None = None) -> None:
    '''Carga un archivo, reconstruye y guarda.'''
    device = device or next(model.parameters()).device
    audio = _load_audio(input_path, sample_rate).to(device)
    T = audio.size(-1)
    T_trim = T - (T % model.total_stride)
    if T_trim != T:
        log.info(f'recortando {T - T_trim} muestras para alinear con ' f'total_stride={model.total_stride}')
    audio = audio[..., :T_trim]
    x = audio.unsqueeze(0) if audio.dim() == 2 else audio
    x_hat = model.reconstruct(x)
    peak = x_hat.abs().max().clamp(min=1e-6)
    x_hat = x_hat / peak
    _save_audio(output_path, x_hat, sample_rate)
    log.info(f'reconstrucción guardada en {output_path}')


@torch.no_grad()
def sample_prior(model: RAVE, output_path: str | Path, duration_seconds: float, sample_rate: int = 16_000, device: torch.device | None = None, seed: int | None = None) -> None:
    '''Muestrea z ~ N(0, I) y decodifica.

    La duración se ajusta al múltiplo más cercano de total_stride /
    sample_rate. Con corpus pequeño y beta bajo el resultado suele ser
    perceptualmente pobre; se documenta pero se ofrece.
    '''
    device = device or next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)
    target_samples = int(duration_seconds * sample_rate)
    target_samples -= target_samples % model.total_stride
    if target_samples <= 0:
        raise ValueError(f'duration_seconds={duration_seconds} demasiado corta para ' f'total_stride={model.total_stride} a {sample_rate} Hz')
    L_z = target_samples // model.total_stride
    audio = model.sample_prior(batch_size=1, length=L_z, device=device)
    peak = audio.abs().max().clamp(min=1e-6)
    audio = audio / peak
    _save_audio(output_path, audio, sample_rate)
    log.info(f'muestra del prior guardada en {output_path}  ' f'(duración: {target_samples / sample_rate:.3f}s)')
