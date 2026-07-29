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

import datetime
import math
import pathlib
import praxis.log as log
import soundfile as sf
import torch
import torch.nn.functional as F


def _load_audio(path: pathlib.Path, target_sr: int) -> torch.Tensor:
    '''Lee un archivo de audio, lo convierte a mono, resamplea y normaliza.

    Args:
        path: Path al archivo de audio.
        target_sr: Frecuencia de muestreo objetivo en Hz.

    Returns:
        Tensor (1, T) en float32, normalizado por el pico absoluto.
    '''

    data, sr = sf.read(path, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).contiguous()

    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        n_out = int(waveform.size(-1) * target_sr / sr)
        waveform = F.interpolate(waveform.unsqueeze(0), size=n_out, mode='linear', align_corners=False).squeeze(0)

    peak = waveform.abs().max().clamp(min=1e-6)
    return waveform / peak


def _save_audio(path: pathlib.Path, waveform: torch.Tensor, sample_rate: int) -> None:
    '''Guarda un tensor como archivo WAV, creando directorios si faltan.

    Args:
        path: Path de salida.
        waveform: Tensor de audio, acepta formas (T), (1, T) o (1, 1, T).
        sample_rate: Frecuencia de muestreo en Hz.
    '''

    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 2:
        waveform = waveform.squeeze(0)

    sf.write(path, waveform.cpu().numpy(), sample_rate)


def describe_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    '''Lee metadatos de un checkpoint sin instanciar el modelo.

    Args:
        checkpoint_path: ruta al archivo .pt.

    Returns:
        Dict con información del archivo, paso de entrenamiento, arquitectura
        derivada, parámetros totales y la configuración original.
    '''

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {}) or {}
    model_state = ckpt.get('model', {}) or {}

    total_params = sum(v.numel() for v in model_state.values() if isinstance(v, torch.Tensor))
    n_bands = config.get('n_bands', 16)
    strides = tuple(config.get('strides', (2, 4, 2)))
    total_stride = n_bands * math.prod(strides)
    sr = config.get('sample_rate', 16_000)
    seg = config.get('segment_length', 0)
    hop = config.get('hop_length') or (seg // 2 if seg else 0)

    return {
        'path': checkpoint_path,
        'size_mb': checkpoint_path.stat().st_size / (1024**2),
        'mtime': datetime.datetime.fromtimestamp(checkpoint_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        'step': ckpt.get('step'),
        'has_optimizer_state': bool(ckpt.get('optimizer')),
        'total_parameters': total_params,
        'total_stride': total_stride,
        'latent_frame_ms': (total_stride / sr) * 1000 if sr else None,
        'segment_seconds': seg / sr if sr else None,
        'hop_seconds': hop / sr if sr else None,
        'config': config,
    }


def load_model(checkpoint_path: pathlib.Path, device: torch.device | None = None) -> tuple[RAVE, dict]:
    '''Carga un checkpoint y reconstruye el modelo con su configuración.

    Args:
        checkpoint_path: Path al archivo .pt guardado durante el entrenamiento.
        device: Dispositivo destino. Si es None, usa CUDA si está disponible,
            en su defecto CPU.

    Returns:
        Tupla (modelo, config) donde modelo está en modo eval y config es el
        diccionario de hiperparámetros con que se entrenó.
    '''

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = ckpt['config']
    model = RAVE(n_bands=config['n_bands'], pqmf_taps=config['pqmf_taps'], hidden_channels=config['hidden_channels'], strides=tuple(config['strides']), latent_dim=config['latent_dim'], n_res_per_block=config['n_res_per_block']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    return model, config


def print_checkpoint_info(info: dict) -> None:
    '''Imprime los metadatos de un checkpoint en formato legible.

    Args:
        info: dict devuelto por describe_checkpoint.
    '''
    c = info['config']
    log.info(f'Checkpoint: {info["path"]}')
    log.info('─' * 64)
    log.info('Archivo')
    log.info(f'\ttamaño         {info["size_mb"]:.2f} MB')
    log.info(f'\tmodificado     {info["mtime"]}')
    log.info('Estado de entrenamiento')
    step = info['step']
    log.info(f'\tpaso           {step:,}' if step is not None else '    paso           (no registrado)')
    log.info(f'\toptimizer      {"presente" if info["has_optimizer_state"] else "ausente"}')
    log.info('Arquitectura')

    for k in ('n_bands', 'pqmf_taps', 'hidden_channels', 'strides', 'latent_dim', 'n_res_per_block'):
        if k in c:
            log.info(f'\t{k:<18} {c[k]}')

    log.info(f'\ttotal_stride       {info["total_stride"]}')
    log.info(f'\tparámetros         {info["total_parameters"]:,}')

    if info['latent_frame_ms'] is not None:
        log.info(f'\tframe latente      {info["latent_frame_ms"]:.2f} ms (@ {c.get("sample_rate")} Hz)')

    log.info('Entrenamiento')

    for k in ('phase', 'sample_rate', 'batch_size', 'lr', 'beta_max', 'warmup_steps', 'free_bits', 'grad_clip', 'stft_weight', 'waveform_weight', 'fft_sizes'):
        if k in c:
            log.info(f'\t{k:<18} {c[k]}')

    if c.get('phase') == 2:
        log.info('Adversarial')
        for k in ('freeze_encoder', 'd_lr', 'adversarial_weight', 'fm_weight', 'd_grad_clip'):
            if k in c:
                log.info(f'\t{k:<18} {c[k]}')

    if info['segment_seconds']:
        log.info(f'\tsegment_length     {c.get("segment_length")} ({info["segment_seconds"]:.3f} s)')

    if info['hop_seconds']:
        log.info(f'\thop_length         {c.get("hop_length") or "auto"} ({info["hop_seconds"]:.3f} s)')

    if 'audio_paths' in c:
        log.info('Datos')

        for p in c['audio_paths']:
            log.info(f'\t{p}')


@torch.no_grad()
def reconstruct(
    model: RAVE,
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    sample_rate: int = 16_000,
    device: torch.device | None = None,
) -> None:
    '''Codifica un archivo con el encoder y lo decodifica de vuelta a audio.

    La longitud del audio se recorta al múltiplo más cercano de
    model.total_stride. La salida se normaliza por su pico absoluto antes de
    guardarse.

    Args:
        model: Modelo RAVE cargado.
        input_path: Path del audio de entrada.
        output_path: Path del audio reconstruido.
        sample_rate: Frecuencia de muestreo en Hz para carga y guardado.
        device: Dispositivo de cómputo. Si es None, usa el del modelo.
    '''

    device = device or next(model.parameters()).device
    audio = _load_audio(input_path, sample_rate).to(device)
    T = audio.size(-1)
    T_trim = T - (T % model.total_stride)

    if T_trim != T:
        log.info(f'Recortando {T - T_trim} muestras para alinear con total_stride={model.total_stride}')

    audio = audio[..., :T_trim]
    x = audio.unsqueeze(0) if audio.dim() == 2 else audio
    x_hat = model.reconstruct(x)
    peak = x_hat.abs().max().clamp(min=1e-6)
    x_hat = x_hat / peak
    _save_audio(output_path, x_hat, sample_rate)
    log.info(f'Reconstrucción guardada en {output_path}')


@torch.no_grad()
def sample_prior(
    model: RAVE,
    output_path: pathlib.Path,
    duration_seconds: float,
    sample_rate: int = 16_000,
    device: torch.device | None = None,
    seed: int | None = None,
) -> None:
    '''Muestrea z desde el prior N(0, I) y decodifica a audio.

    La duración se ajusta al múltiplo más cercano de total_stride /
    sample_rate. Con corpus pequeño y beta bajo el resultado suele ser
    perceptualmente pobre porque el latente entrenado ocupa una variedad
    estrecha dentro de N(0, I).

    Args:
        model: Modelo RAVE cargado.
        output_path: Path del audio generado.
        duration_seconds: Duración objetivo en segundos.
        sample_rate: Frecuencia de muestreo en Hz.
        device: Dispositivo de cómputo. Si es None, usa el del modelo.
        seed: Semilla para reproducibilidad. Si es None, cada llamada produce
            audio distinto.

    Raises:
        ValueError: Si duration_seconds resulta en menos de total_stride
            muestras a la frecuencia dada.
    '''

    device = device or next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)

    target_samples = int(duration_seconds * sample_rate)
    target_samples -= target_samples % model.total_stride

    if target_samples <= 0:
        raise ValueError(f'Duration_seconds={duration_seconds} demasiado corta para total_stride={model.total_stride} a {sample_rate} Hz')

    L_z = target_samples // model.total_stride
    audio = model.sample_prior(batch_size=1, length=L_z, device=device)
    peak = audio.abs().max().clamp(min=1e-6)
    audio = audio / peak

    _save_audio(output_path, audio, sample_rate)
    log.info(f'Muestra del prior guardada en {output_path} (Duración: {target_samples / sample_rate:.2f}s)')
