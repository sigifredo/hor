'''CLI del subpaquete hor.rave con soporte de fases 1 y 2.

Uso:

    # Fase 1 (VAE puro con anti-colapso):
    python -m rave train --phase 1 --audio archivo.wav --out-dir runs/rave_v1

    # Fase 2 (adversarial) reanudando desde fase 1:
    python -m rave train --phase 2 --audio archivo.wav \\
        --out-dir runs/rave_v1_gan \\
        --resume-from runs/rave_v1/checkpoints/rave_p1_final.pt \\
        --resume-mode finetune

    # Reconstrucción con cualquier checkpoint:
    python -m rave reconstruct --checkpoint runs/.../rave_p2_final.pt \\
        --audio entrada.wav --output reconstruido.wav

    # Muestra del prior (solo útil si la KL no está colapsada):
    python -m rave sample --checkpoint runs/.../rave_p2_final.pt \\
        --duration 4.0 --output muestra_prior.wav

    # Info de un checkpoint:
    python -m rave info --checkpoint runs/.../rave_p2_final.pt
'''

from __future__ import annotations
from . import engine
from . import generate

import argparse
import pathlib
import praxis
import praxis.log as log
import time


def _add_train_args(sp: argparse.ArgumentParser) -> None:
    # datos y segmentación
    sp.add_argument('--audio', nargs='+', required=True, help='uno o más archivos de audio para entrenar')
    sp.add_argument('--out-dir', type=pathlib.Path, required=True, help='carpeta donde se guardan checkpoints y logs')
    sp.add_argument('--sample-rate', type=int, default=16_000, help='frecuencia de muestreo objetivo en Hz')
    sp.add_argument('--segment-length', type=int, default=32_768, help='muestras por segmento; a 16 kHz 32768 son 2 s; debe ser múltiplo del total_stride del modelo (256 con los defaults)')
    sp.add_argument('--hop-length', type=int, default=None, help='paso entre segmentos consecutivos en muestras; por defecto segment_length / 2')
    sp.add_argument('--val-fraction', type=float, default=0.1, help='fracción de segmentos reservados para validación')
    sp.add_argument('--batch-size', type=int, default=4, help='segmentos por paso; con 6 GB VRAM y fase 2, dejar en 4')
    sp.add_argument('--num-workers', type=int, default=2, help='procesos paralelos del DataLoader')

    # arquitectura del modelo
    sp.add_argument('--n-bands', type=int, default=16, help='número de subbandas del PQMF')
    sp.add_argument('--pqmf-taps', type=int, default=126, help='taps del prototipo PQMF, debe ser par')
    sp.add_argument('--hidden-channels', type=int, default=64, help='canales del primer bloque del encoder')
    sp.add_argument('--strides', type=int, nargs='+', default=[2, 4, 2], help='factores de diezmado por bloque del encoder')
    sp.add_argument('--latent-dim', type=int, default=64, help='dimensión del espacio latente')
    sp.add_argument('--n-res-per-block', type=int, default=3, help='unidades residuales por bloque')

    # pérdidas VAE
    sp.add_argument('--fft-sizes', type=int, nargs='+', default=[512, 1024, 2048], help='escalas FFT para la pérdida STFT multi-escala')
    sp.add_argument('--stft-weight', type=float, default=1.0, help='peso del término espectral')
    sp.add_argument('--waveform-weight', type=float, default=0.1, help='peso del L1 sobre la forma de onda')
    sp.add_argument('--beta-max', type=float, default=1e-4, help='peso final del término KL tras el warmup; bajado desde 0.1 para prevenir colapso posterior')
    sp.add_argument('--warmup-steps', type=int, default=100_000, help='pasos del warmup lineal para el KL')
    sp.add_argument('--free-bits', type=float, default=0.1, help='piso en nats por dimensión latente para la KL; 0 desactiva')

    # optimización G
    sp.add_argument('--lr', type=float, default=1e-4, help='learning rate de Adam para el generador')
    sp.add_argument('--grad-clip', type=float, default=1.0, help='umbral para clipping de gradiente del generador')
    sp.add_argument('--n-steps', type=int, default=100_000, help='total de pasos de entrenamiento en esta corrida')

    # fase y adversarial
    sp.add_argument('--phase', type=int, default=1, choices=[1, 2], help='fase de entrenamiento; 1 = VAE puro, 2 = adversarial')
    sp.add_argument('--freeze-encoder', action='store_true', default=True, help='en fase 2, congelar encoder y PQMF (default: True, siguiendo RAVE original)')
    sp.add_argument('--no-freeze-encoder', dest='freeze_encoder', action='store_false', help='permitir que el encoder siga aprendiendo en fase 2')
    sp.add_argument('--d-lr', type=float, default=1e-4, help='learning rate del discriminador')
    sp.add_argument('--adversarial-weight', type=float, default=1.0, help='peso del término hinge adversarial en la pérdida del generador')
    sp.add_argument('--fm-weight', type=float, default=10.0, help='peso del feature matching en la pérdida del generador')
    sp.add_argument('--d-grad-clip', type=float, default=1.0, help='umbral para clipping de gradiente del discriminador')

    # loop y checkpoints
    sp.add_argument('--log-every', type=int, default=100, help='pasos entre líneas de log en stdout y CSV')
    sp.add_argument('--val-every', type=int, default=1_000, help='pasos entre corridas de validación')
    sp.add_argument('--ckpt-every', type=int, default=5_000, help='pasos entre guardado de checkpoints')
    sp.add_argument('--keep-last-ckpts', type=int, default=3, help='cuántos checkpoints periódicos retener')
    sp.add_argument('--seed', type=int, default=42, help='semilla para reproducibilidad')
    sp.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='dispositivo de cómputo')

    # reanudar/finetune
    sp.add_argument('--resume-from', type=pathlib.Path, default=None, help='ruta a un checkpoint desde el cual reanudar')
    sp.add_argument('--resume-mode', choices=['continue', 'finetune'], default='continue', help='continue: restaura pesos, optimizadores y step. finetune: solo pesos; útil para pasar de fase 1 a fase 2')


def _run_info(args: argparse.Namespace) -> None:
    generate.print_checkpoint_info(generate.describe_checkpoint(args.checkpoint))


def _run_reconstruct(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    device = engine.select_device(args.device)
    model, config = generate.load_model(args.checkpoint, device)
    generate.reconstruct(
        model,
        args.audio,
        args.output,
        sample_rate=config['sample_rate'],
        device=device,
    )
    log.info(f'Duración de generación: {praxis.time.seconds_to_hms(time.perf_counter() - start_time)}')


def _run_sample(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    device = engine.select_device(args.device)
    model, config = generate.load_model(args.checkpoint, device)
    generate.sample_prior(
        model,
        args.output,
        duration_seconds=args.duration,
        sample_rate=config['sample_rate'],
        device=device,
        seed=args.seed,
    )
    log.info(f'Duración de generación: {praxis.time.seconds_to_hms(time.perf_counter() - start_time)}')


def _run_train(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    config = engine.TrainConfig(
        audio_paths=list(args.audio),
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
        segment_length=args.segment_length,
        hop_length=args.hop_length,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_bands=args.n_bands,
        pqmf_taps=args.pqmf_taps,
        hidden_channels=args.hidden_channels,
        strides=tuple(args.strides),
        latent_dim=args.latent_dim,
        n_res_per_block=args.n_res_per_block,
        fft_sizes=tuple(args.fft_sizes),
        beta_max=args.beta_max,
        warmup_steps=args.warmup_steps,
        stft_weight=args.stft_weight,
        waveform_weight=args.waveform_weight,
        free_bits=args.free_bits,
        lr=args.lr,
        grad_clip=args.grad_clip,
        n_steps=args.n_steps,
        phase=args.phase,
        freeze_encoder=args.freeze_encoder,
        d_lr=args.d_lr,
        adversarial_weight=args.adversarial_weight,
        fm_weight=args.fm_weight,
        d_grad_clip=args.d_grad_clip,
        log_every=args.log_every,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        keep_last_ckpts=args.keep_last_ckpts,
        seed=args.seed,
        resume_from=args.resume_from,
        resume_mode=args.resume_mode,
        device=args.device,
    )
    engine.train(config)
    log.info(f'Duración del entrenamiento: {praxis.time.seconds_to_hms(time.perf_counter() - start_time)}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rave')
    sub = parser.add_subparsers(dest='command', required=True)

    train_p = sub.add_parser('train', help='entrenar el VAE (fase 1 o 2)')
    _add_train_args(train_p)

    rec_p = sub.add_parser('reconstruct', help='reconstruir un archivo')
    rec_p.add_argument('--checkpoint', type=pathlib.Path, required=True)
    rec_p.add_argument('--audio', type=pathlib.Path, required=True)
    rec_p.add_argument('--output', type=pathlib.Path, required=True)
    rec_p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    sam_p = sub.add_parser('sample', help='muestrear desde el prior')
    sam_p.add_argument('--checkpoint', type=pathlib.Path, required=True)
    sam_p.add_argument('--output', type=pathlib.Path, required=True)
    sam_p.add_argument('--duration', type=float, required=True)
    sam_p.add_argument('--seed', type=int, default=None)
    sam_p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])

    info_p = sub.add_parser('info', help='inspeccionar metadatos de un checkpoint')
    info_p.add_argument('--checkpoint', type=pathlib.Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == 'train':
        _run_train(args)
    elif args.command == 'reconstruct':
        _run_reconstruct(args)
    elif args.command == 'sample':
        _run_sample(args)
    elif args.command == 'info':
        _run_info(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log.error('Algoritmo interrumpido por el usuario')
        raise SystemExit(130)
