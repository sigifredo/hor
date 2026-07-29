'''Loop de entrenamiento del VAE tipo RAVE en dos fases.

Fase 1 (VAE puro):
    * Loss: STFT multi-escala + L1 forma de onda + KL con free bits.
    * β_max bajo (1e-4 default) con warmup largo (100k pasos).
    * Objetivo: latente no colapsado con reconstrucción razonablemente buena.

Fase 2 (adversarial):
    * Añade discriminador multi-escala tipo MelGAN.
    * Hinge loss para D, hinge + feature matching para G.
    * Encoder opcionalmente frozen (default: sí, siguiendo RAVE original).
    * Objetivo: cerrar la brecha perceptual eliminando la borrosidad espectral.

Diseño:
    * Iteración por pasos, no por épocas.
    * Optimizadores separados para G y D en fase 2, con lr y betas propios.
    * Gradient clipping por norma en ambos.
    * Alternancia: en cada paso se actualiza D primero (con x_hat detached),
      luego G con adversarial + feature matching + reconstrucción + KL.
    * Checkpoints periódicos con retención de los últimos K.
    * Logging estructurado a CSV, campos distintos por fase.
    * Reanudación en modos continue y finetune. En fase 2 se puede cargar un
      checkpoint de fase 1 con --resume-mode finetune para arrancar el
      adversarial desde el VAE ya entrenado.
'''

from __future__ import annotations
from .data import AudioSegmentDataset, split_train_val
from .discriminator import MultiScaleDiscriminator
from .losses import RAVELoss, hinge_loss_d, hinge_loss_g, feature_matching_loss
from .model import RAVE

import csv
import dataclasses
import pathlib
import praxis
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
    batch_size: int = 4
    num_workers: int = 2

    # modelo
    n_bands: int = 16
    pqmf_taps: int = 126
    hidden_channels: int = 64
    strides: tuple[int, ...] = (2, 4, 2)
    latent_dim: int = 64
    n_res_per_block: int = 3

    # pérdidas VAE
    fft_sizes: tuple[int, ...] = (512, 1024, 2048)
    beta_max: float = 1e-4
    warmup_steps: int = 100_000
    stft_weight: float = 1.0
    waveform_weight: float = 0.1
    free_bits: float = 0.1

    # optimización G
    lr: float = 1e-4
    adam_betas: tuple[float, float] = (0.5, 0.9)
    grad_clip: float = 1.0

    # fase y adversarial
    phase: int = 1
    freeze_encoder: bool = True
    d_lr: float = 1e-4
    d_adam_betas: tuple[float, float] = (0.5, 0.9)
    adversarial_weight: float = 1.0
    fm_weight: float = 10.0
    d_grad_clip: float = 1.0

    # loop
    n_steps: int = 100_000
    log_every: int = 100
    val_every: int = 1_000
    ckpt_every: int = 5_000
    keep_last_ckpts: int = 3
    seed: int = 42
    prune_old_checkpoints: bool = False

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
    optimizer_g: torch.optim.Optimizer,
    discriminator: MultiScaleDiscriminator | None,
    optimizer_d: torch.optim.Optimizer | None,
    mode: str,
) -> int:
    '''Restaura estado de modelo, discriminador y optimizadores.

    Modos:
        continue: restaura todo, incluyendo step.
        finetune: solo pesos del modelo (y del D si existe en el checkpoint),
            optimizadores y step arrancan desde cero. Útil para pasar de
            fase 1 a fase 2.

    Args:
        path: ruta al checkpoint a cargar.
        model: modelo destino.
        optimizer_g: optimizador del generador.
        discriminator: discriminador destino, opcional.
        optimizer_d: optimizador del D, opcional.
        mode: 'continue' o 'finetune'.

    Returns:
        Step inicial.
    '''

    if mode not in ('continue', 'finetune'):
        raise ValueError(f'resume_mode inválido: {mode}')

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])

    if discriminator is not None and 'discriminator' in ckpt:
        discriminator.load_state_dict(ckpt['discriminator'])

    if mode == 'continue':
        optimizer_g.load_state_dict(ckpt['optimizer'])
        if optimizer_d is not None and 'optimizer_d' in ckpt:
            optimizer_d.load_state_dict(ckpt['optimizer_d'])
        start_step = int(ckpt.get('step', 0))
        log.info(f'reanudando desde step {start_step:,} (optimizadores restaurados)')
        return start_step

    log.info('fine-tuning desde pesos del checkpoint (optimizadores y step desde cero)')
    return 0


def _prune_old_checkpoints(ckpt_dir: pathlib.Path, pattern: str, keep_last: int) -> None:
    ckpts = sorted(ckpt_dir.glob(pattern))
    for old in ckpts[:-keep_last]:
        old.unlink()


def _save_checkpoint(
    path: pathlib.Path,
    model: RAVE,
    optimizer_g: torch.optim.Optimizer,
    step: int,
    config: TrainConfig,
    discriminator: MultiScaleDiscriminator | None = None,
    optimizer_d: torch.optim.Optimizer | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'model': model.state_dict(),
        'optimizer': optimizer_g.state_dict(),
        'step': step,
        'config': config.__dict__,
    }
    if discriminator is not None:
        payload['discriminator'] = discriminator.state_dict()
    if optimizer_d is not None:
        payload['optimizer_d'] = optimizer_d.state_dict()
    torch.save(payload, path)


@torch.no_grad()
def _validate_phase1(
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
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


@torch.no_grad()
def _validate_phase2(
    model: RAVE,
    discriminator: MultiScaleDiscriminator,
    loss_fn: RAVELoss,
    loader: torch.utils.data.DataLoader,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    discriminator.eval()
    sums: dict[str, float] = {}
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        x_hat, mu, log_sigma = model(batch)
        rec_losses = loss_fn(batch, x_hat, mu, log_sigma, step)

        d_out_real = discriminator(batch)
        d_out_fake = discriminator(x_hat)
        features_real = [f for f, _ in d_out_real]
        features_fake = [f for f, _ in d_out_fake]
        logits_real = [l for _, l in d_out_real]
        logits_fake = [l for _, l in d_out_fake]

        d_loss = hinge_loss_d(logits_real, logits_fake)
        g_adv = hinge_loss_g(logits_fake)
        fm = feature_matching_loss(features_real, features_fake)

        agg = {
            **rec_losses,
            'd_loss': d_loss,
            'g_adv': g_adv,
            'fm': fm,
        }

        for k, v in agg.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())

        n_batches += 1

    model.train()
    discriminator.train()
    return {k: v / max(n_batches, 1) for k, v in sums.items()}


def select_device(name: str | torch.device | None) -> torch.device:
    if isinstance(name, torch.device):
        return name

    if name is None or name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.device(name)


def _build_dataset(config: TrainConfig):
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
    return dataset, train_set, val_set, train_loader, val_loader


def _build_model(config: TrainConfig, device: torch.device) -> RAVE:
    model = RAVE(
        n_bands=config.n_bands,
        pqmf_taps=config.pqmf_taps,
        hidden_channels=config.hidden_channels,
        strides=config.strides,
        latent_dim=config.latent_dim,
        n_res_per_block=config.n_res_per_block,
    ).to(device)

    if config.segment_length % model.total_stride != 0:
        raise ValueError(f'segment_length={config.segment_length} no es múltiplo de total_stride={model.total_stride}')

    return model


def _build_loss(config: TrainConfig, device: torch.device) -> RAVELoss:
    return RAVELoss(
        fft_sizes=config.fft_sizes,
        beta_max=config.beta_max,
        warmup_steps=config.warmup_steps,
        stft_weight=config.stft_weight,
        waveform_weight=config.waveform_weight,
        free_bits=config.free_bits,
    ).to(device)


def _train_phase1(config: TrainConfig) -> pathlib.Path:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    dataset, train_set, val_set, train_loader, val_loader = _build_dataset(config)
    model = _build_model(config, device)
    loss_fn = _build_loss(config, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.adam_betas)

    start_step = 0
    if config.resume_from is not None:
        start_step = _load_resume(config.resume_from, model, optimizer, None, None, config.resume_mode)

    train_iter = _infinite(train_loader)
    log_path = config.out_dir / 'train_log.csv'
    val_path = config.out_dir / 'val_log.csv'
    log_fields = ['step', 'total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'kl_raw', 'beta', 'elapsed_s']
    val_fields = ['step'] + [f'val_{k}' for k in ('total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'kl_raw')]

    if not log_path.exists():
        with log_path.open('w', newline='') as f:
            csv.writer(f).writerow(log_fields)

    if not val_path.exists():
        with val_path.open('w', newline='') as f:
            csv.writer(f).writerow(val_fields)

    param_counts = model.parameter_count()
    log.info(f'[fase 1] segmentos totales: {len(dataset)} (train: {len(train_set)}, val: {len(val_set)})')
    log.info(f'[fase 1] parámetros: encoder={param_counts["encoder"]:,}, decoder={param_counts["decoder"]:,}, total={param_counts["total"]:,}')
    log.info(f'[fase 1] device: {device}, total_stride: {model.total_stride}, β_max={config.beta_max}, warmup={config.warmup_steps}, free_bits={config.free_bits}')

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

        for k in ('total', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'kl_raw', 'beta'):
            running[k] = running.get(k, 0.0) + float(losses[k].item())

        if step % config.log_every == 0:
            elapsed = praxis.time.seconds_to_hms(time.time() - start_time)
            avg = {k: v / config.log_every for k, v in running.items()}
            log.info(f'[fase 1] step {step:>7d}  total={avg["total"]:.4f}  sc={avg["stft_sc"]:.4f}  lmag={avg["stft_log_mag"]:.4f}  l1={avg["waveform_l1"]:.4f}  kl={avg["kl"]:.4f}  kl_raw={avg["kl_raw"]:.4f}  β={avg["beta"]:.6f}  elapsed=[{elapsed}]')
            with log_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, avg['total'], avg['stft_sc'], avg['stft_log_mag'], avg['waveform_l1'], avg['kl'], avg['kl_raw'], avg['beta'], f'{elapsed}'])
            running = {}

        if step % config.val_every == 0 and len(val_set) > 0:
            val = _validate_phase1(model, loss_fn, val_loader, step, device)
            log.info(f'  [val] total={val["total"]:.4f}  sc={val["stft_sc"]:.4f}  kl={val["kl"]:.4f}  kl_raw={val["kl_raw"]:.4f}')
            with val_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, val['total'], val['stft_sc'], val['stft_log_mag'], val['waveform_l1'], val['kl'], val['kl_raw']])

        if step % config.ckpt_every == 0:
            ckpt_path = ckpt_dir / f'rave_p1_step{step:08d}.pt'
            _save_checkpoint(ckpt_path, model, optimizer, step, config)

            if config.prune_old_checkpoints:
                _prune_old_checkpoints(ckpt_dir, 'rave_p1_step*.pt', config.keep_last_ckpts)

            log.info(f'  checkpoint guardado: {ckpt_path.name}')

    final_step = start_step + config.n_steps
    final_path = ckpt_dir / 'rave_p1_final.pt'
    _save_checkpoint(final_path, model, optimizer, final_step, config)
    log.info(f'[fase 1] entrenamiento completo. checkpoint final: {final_path}')
    return final_path


def _train_phase2(config: TrainConfig) -> pathlib.Path:
    torch.manual_seed(config.seed)
    device = select_device(config.device)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = config.out_dir / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)

    dataset, train_set, val_set, train_loader, val_loader = _build_dataset(config)
    model = _build_model(config, device)
    loss_fn = _build_loss(config, device)

    discriminator = MultiScaleDiscriminator().to(device)

    if config.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.pqmf.parameters():
            p.requires_grad = False

    g_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_g = torch.optim.Adam(g_params, lr=config.lr, betas=config.adam_betas)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.d_lr, betas=config.d_adam_betas)

    start_step = 0
    if config.resume_from is not None:
        start_step = _load_resume(config.resume_from, model, optimizer_g, discriminator, optimizer_d, config.resume_mode)

    train_iter = _infinite(train_loader)
    log_path = config.out_dir / 'train_log.csv'
    val_path = config.out_dir / 'val_log.csv'
    log_fields = ['step', 'total_g', 'stft_sc', 'stft_log_mag', 'waveform_l1', 'kl', 'kl_raw', 'beta', 'g_adv', 'fm', 'd_loss', 'elapsed_s']
    val_fields = ['step', 'val_total', 'val_stft_sc', 'val_stft_log_mag', 'val_waveform_l1', 'val_kl', 'val_kl_raw', 'val_g_adv', 'val_fm', 'val_d_loss']

    if not log_path.exists():
        with log_path.open('w', newline='') as f:
            csv.writer(f).writerow(log_fields)
    if not val_path.exists():
        with val_path.open('w', newline='') as f:
            csv.writer(f).writerow(val_fields)

    param_counts = model.parameter_count()
    n_d_params = sum(p.numel() for p in discriminator.parameters())
    log.info(f'[fase 2] segmentos totales: {len(dataset)} (train: {len(train_set)}, val: {len(val_set)})')
    log.info(f'[fase 2] G params: {param_counts["total"]:,} (entrenables: {sum(p.numel() for p in g_params):,})')
    log.info(f'[fase 2] D params: {n_d_params:,}')
    log.info(f'[fase 2] device: {device}, adversarial_weight={config.adversarial_weight}, fm_weight={config.fm_weight}, freeze_encoder={config.freeze_encoder}')

    model.train()
    discriminator.train()
    start_time = time.time()
    running: dict[str, float] = {}

    for step in range(start_step + 1, start_step + config.n_steps + 1):
        batch = next(train_iter).to(device, non_blocking=True)

        # actualizar D
        with torch.no_grad():
            x_hat_detached, _, _ = model(batch)
        d_out_real = discriminator(batch)
        d_out_fake = discriminator(x_hat_detached)
        logits_real = [l for _, l in d_out_real]
        logits_fake = [l for _, l in d_out_fake]
        d_loss = hinge_loss_d(logits_real, logits_fake)

        optimizer_d.zero_grad(set_to_none=True)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.d_grad_clip)
        optimizer_d.step()

        # actualizar G
        x_hat, mu, log_sigma = model(batch)
        rec_losses = loss_fn(batch, x_hat, mu, log_sigma, step)

        d_out_real_g = discriminator(batch)
        d_out_fake_g = discriminator(x_hat)
        features_real = [f for f, _ in d_out_real_g]
        features_fake = [f for f, _ in d_out_fake_g]
        logits_fake_g = [l for _, l in d_out_fake_g]

        g_adv = hinge_loss_g(logits_fake_g)
        fm = feature_matching_loss(features_real, features_fake)

        total_g = rec_losses['total'] + config.adversarial_weight * g_adv + config.fm_weight * fm

        optimizer_g.zero_grad(set_to_none=True)
        total_g.backward()
        torch.nn.utils.clip_grad_norm_(g_params, config.grad_clip)
        optimizer_g.step()

        step_metrics = {
            'total_g': float(total_g.item()),
            'stft_sc': float(rec_losses['stft_sc'].item()),
            'stft_log_mag': float(rec_losses['stft_log_mag'].item()),
            'waveform_l1': float(rec_losses['waveform_l1'].item()),
            'kl': float(rec_losses['kl'].item()),
            'kl_raw': float(rec_losses['kl_raw'].item()),
            'beta': float(rec_losses['beta'].item()),
            'g_adv': float(g_adv.item()),
            'fm': float(fm.item()),
            'd_loss': float(d_loss.item()),
        }
        for k, v in step_metrics.items():
            running[k] = running.get(k, 0.0) + v

        if step % config.log_every == 0:
            elapsed = time.time() - start_time
            avg = {k: v / config.log_every for k, v in running.items()}
            log.info(f'[fase 2] step {step:>7d}  G={avg["total_g"]:.4f}  sc={avg["stft_sc"]:.4f}  l1={avg["waveform_l1"]:.4f}  kl={avg["kl"]:.4f}  adv={avg["g_adv"]:.4f}  fm={avg["fm"]:.4f}  D={avg["d_loss"]:.4f}')
            with log_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, avg['total_g'], avg['stft_sc'], avg['stft_log_mag'], avg['waveform_l1'], avg['kl'], avg['kl_raw'], avg['beta'], avg['g_adv'], avg['fm'], avg['d_loss'], f'{elapsed:.1f}'])
            running = {}

        if step % config.val_every == 0 and len(val_set) > 0:
            val = _validate_phase2(model, discriminator, loss_fn, val_loader, step, device)
            log.info(f'  [val] total={val["total"]:.4f}  sc={val["stft_sc"]:.4f}  kl={val["kl"]:.4f}  adv={val["g_adv"]:.4f}  fm={val["fm"]:.4f}  D={val["d_loss"]:.4f}')
            with val_path.open('a', newline='') as f:
                csv.writer(f).writerow([step, val['total'], val['stft_sc'], val['stft_log_mag'], val['waveform_l1'], val['kl'], val['kl_raw'], val['g_adv'], val['fm'], val['d_loss']])

        if step % config.ckpt_every == 0:
            ckpt_path = ckpt_dir / f'rave_p2_step{step:08d}.pt'
            _save_checkpoint(ckpt_path, model, optimizer_g, step, config, discriminator, optimizer_d)
            _prune_old_checkpoints(ckpt_dir, 'rave_p2_step*.pt', config.keep_last_ckpts)
            log.info(f'  checkpoint guardado: {ckpt_path.name}')

    final_step = start_step + config.n_steps
    final_path = ckpt_dir / 'rave_p2_final.pt'
    _save_checkpoint(final_path, model, optimizer_g, final_step, config, discriminator, optimizer_d)
    log.info(f'[fase 2] entrenamiento completo. checkpoint final: {final_path}')
    return final_path


def train(config: TrainConfig) -> pathlib.Path:
    '''Despacha a la fase correspondiente según config.phase.'''
    if config.phase == 1:
        return _train_phase1(config)
    if config.phase == 2:
        return _train_phase2(config)
    raise ValueError(f'phase inválida: {config.phase} (debe ser 1 o 2)')
