'''Loop de entrenamiento del VAE tipo RAVE.

Diseño:
    * Iteración por pasos (no por épocas): más flexible para corpora pequeños
      donde una época dura pocos segundos y el warmup del KL se define en
      pasos globales.
    * Optimizador Adam con betas=(0.5, 0.9), estándar para modelos
      generativos de audio.
    * Gradient clipping por norma con umbral 1.0, protección contra picos
      esporádicos durante el warmup del KL.
    * Checkpoints periódicos con retención de los últimos K (default 3).
    * Logging estructurado a CSV en el directorio de salida, más resumen a
      stdout cada log_every pasos.
    * Validación sobre subset separado del mismo material: útil como señal
      de estabilidad, no como test de generalización real.
    * Reanudación desde checkpoint en dos modos: continue (retomar estado
      completo) y finetune (cargar solo pesos del modelo).
'''

from __future__ import annotations
from .data import AudioSegmentDataset, split_train_val
from .losses import RAVELoss
from .model import RAVE

import csv
import dataclasses
import pathlib
import praxis.log as log
import time
import torch


@dataclasses.dataclass
class TrainConfig:
    audio_paths: list[str] = dataclasses.field(default_factory=list)
    out_dir: pathlib.Path = pathlib.Path('runs/rave')
    sample_rate: int = 16_000
    segment_length: int = 32_768
    hop_length: int | None = None
    val_fraction: float = 0.1
    batch_size: int = 8
    num_workers: int = 2

    # modelo
    n_bands: int = 16
    pqmf_taps: int = 126
    hidden_channels: int = 64
    strides: tuple[int, ...] = (2, 4, 2)
    latent_dim: int = 64
    n_res_per_block: int = 3

    # pérdidas
    fft_sizes: tuple[int, ...] = (512, 1024, 2048)
    beta_max: float = 0.1
    warmup_steps: int = 10_000
    stft_weight: float = 1.0
    waveform_weight: float = 0.1

    # optimización
    lr: float = 1e-4
    adam_betas: tuple[float, float] = (0.5, 0.9)
    grad_clip: float = 1.0

    # loop
    n_steps: int = 100_000
    log_every: int = 100
    val_every: int = 1_000
    ckpt_every: int = 5_000
    keep_last_ckpts: int = 3
    seed: int = 42

    # reanudar/finetune
    resume_from: pathlib.Path | None = None
    resume_mode: str = 'continue'

    device: str = 'auto'


def _infinite(loader: torch.utils.data.DataLoader):
    while True:
        for batch in loader:
            yield batch


def _load_resume(
    path: pathlib.Path,
    model: RAVE,
    optimizer: torch.optim.Optimizer,
    mode: str,
) -> int:
    '''Restaura estado del modelo y opcionalmente del optimizador.

    Args:
        path: ruta al checkpoint a cargar.
        model: modelo destino, debe tener la misma arquitectura que la usada
            al guardar el checkpoint.
        optimizer: optimizador destino.
        mode: 'continue' restaura pesos, optimizer y step del checkpoint;
            'finetune' restaura solo los pesos, optimizer y step arrancan
            desde cero.

    Returns:
        Step inicial. Cero en modo finetune, step guardado en modo continue.

    Raises:
        ValueError: si mode no es 'continue' ni 'finetune'.
    '''

    if mode not in ('continue', 'finetune'):
        raise ValueError(f'resume_mode inválido: {mode}')

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])

    if mode == 'continue':
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = int(ckpt.get('step', 0))
        log.info(f'reanudando desde step {start_step:,} (optimizer restaurado)')
        return start_step

    log.info('fine-tuning desde pesos del checkpoint (optimizer y step desde cero)')
    return 0


def _prune_old_checkpoints(
    ckpt_dir: pathlib.Path,
    pattern: str,
    keep_last: int,
) -> None:
    ckpts = sorted(ckpt_dir.glob(pattern))
    for old in ckpts[:-keep_last]:
        old.unlink()


def _save_checkpoint(
    path: pathlib.Path,
    model: RAVE,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'config': config.__dict__,
        },
        path,
    )


@torch.no_grad()
def _validate(
    model: RAVE,
    loss_fn: RAVELoss,
    loader: torch.utils.data.DataLoader,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x_hat, mu, log_sigma = model(batch)
        losses = loss_fn(batch, x_hat, mu, log_sigma, step)

        for k, v in losses.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())

        n_batches += 1

    model.train()
    return {k: v / n_batches for k, v in sums.items()}


def select_device(name: str | torch.device | None) -> torch.device:
    if isinstance(name, torch.device):
        return name

    if name is None or name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.device(name)


def train(config: TrainConfig) -> pathlib.Path:
    '''Ejecuta el entrenamiento.

    Args:
        config: hiperparámetros del entrenamiento.

    Returns:
        Path del checkpoint final guardado.
    '''

    torch.manual_seed(config.seed)
    device = select_device(config.device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    dataset = AudioSegmentDataset(
        audio_paths=config.audio_paths,
        segment_length=config.segment_length,
        hop_length=config.hop_length,
        sample_rate=config.sample_rate,
        normalize=True,
    )
    train_set, val_set = split_train_val(
        dataset,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )

    model = RAVE(
        n_bands=config.n_bands,
        pqmf_taps=config.pqmf_taps,
        hidden_channels=config.hidden_channels,
        strides=config.strides,
        latent_dim=config.latent_dim,
        n_res_per_block=config.n_res_per_block,
    ).to(device)
    if config.segment_length % model.total_stride != 0:
        raise ValueError(f'segment_length={config.segment_length} no es múltiplo de ' f'total_stride={model.total_stride}')

    loss_fn = RAVELoss(
        fft_sizes=config.fft_sizes,
        beta_max=config.beta_max,
        warmup_steps=config.warmup_steps,
        stft_weight=config.stft_weight,
        waveform_weight=config.waveform_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.adam_betas)

    start_step = 0
    if config.resume_from is not None:
        start_step = _load_resume(config.resume_from, model, optimizer, config.resume_mode)

    train_iter = _infinite(train_loader)
    log_path = config.out_dir / 'train_log.csv'
    val_path = config.out_dir / 'val_log.csv'
    log_fields = ['step', 'total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'beta', 'elapsed_s']
    val_fields = ['step'] + [f'val_{k}' for k in ('total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl')]

    if not log_path.exists():
        with log_path.open('w', newline='') as f:
            csv.writer(f).writerow(log_fields)

    if not val_path.exists():
        with val_path.open('w', newline='') as f:
            csv.writer(f).writerow(val_fields)

    param_counts = model.parameter_count()
    log.info(f'segmentos totales: {len(dataset)} ' f'(train: {len(train_set)}, val: {len(val_set)})')
    log.info(f'parámetros: encoder={param_counts["encoder"]:,}, ' f'decoder={param_counts["decoder"]:,}, ' f'total={param_counts["total"]:,}')
    log.info(f'device: {device}, total_stride: {model.total_stride}')
    log.info(f'segmentos por step (batch): {config.batch_size}, ' f'pasos totales: {config.n_steps}')

    model.train()
    start_time = time.time()
    running: dict[str, float] = {}

    for step in range(start_step + 1, start_step + config.n_steps + 1):
        batch = next(train_iter).to(device, non_blocking=True)
        x_hat, mu, log_sigma = model(batch)
        losses = loss_fn(batch, x_hat, mu, log_sigma, step)
        optimizer.zero_grad(set_to_none=True)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        for k in ('total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'beta'):
            running[k] = running.get(k, 0.0) + float(losses[k].item())

        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            avg = {k: v / config.log_every for k, v in running.items()}
            log.info(f'step {step:>7d}  total={avg["total"]:.4f}  ' f'sc={avg["stft_sc"]:.4f}  ' f'lmag={avg["stft_log_mag"]:.4f}  ' f'l1={avg["waveform_l1"]:.4f}  ' f'kl={avg["kl"]:.4f}  ' f'β={avg["beta"]:.4f}  ' f'[{elapsed:.1f}s]')

            with log_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, avg['total'], avg['stft_sc'], avg['stft_log_mag'], avg['waveform_l1'], avg['kl'], avg['beta'], elapsed])
            running = {}

        if step % config.val_every == 0 and len(val_set) > 0:
            val = _validate(model, loss_fn, val_loader, step, device)
            log.info(f'  [val] total={val["total"]:.4f}  ' f'sc={val["stft_sc"]:.4f}  ' f'lmag={val["stft_log_mag"]:.4f}  ' f'kl={val["kl"]:.4f}')

            with val_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, val['total'], val['stft_sc'], val['stft_log_mag'], val['waveform_l1'], val['kl']])

        if step % config.ckpt_every == 0:
            ckpt_path = ckpt_dir / f'rave_step{step:08d}.pt'
            _save_checkpoint(ckpt_path, model, optimizer, step, config)
            _prune_old_checkpoints(ckpt_dir, 'rave_step*.pt', config.keep_last_ckpts)
            log.info(f'  checkpoint guardado: {ckpt_path.name}')

    final_step = start_step + config.n_steps
    final_path = ckpt_dir / 'rave_final.pt'
    _save_checkpoint(final_path, model, optimizer, final_step, config)
    log.info(f'entrenamiento completo. checkpoint final: {final_path}')

    return final_path
