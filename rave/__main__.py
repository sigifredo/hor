'''CLI del subpaquete hor.rave.

Uso:
    python -m rave train --audio archivo.wav --out-dir runs/rave_v1
    python -m rave reconstruct --checkpoint runs/rave_v1/checkpoints/rave_final.pt \\
        --audio entrada.wav --output reconstruido.wav
    python -m rave sample --checkpoint runs/rave_v1/checkpoints/rave_final.pt \\
        --duration 4.0 --output muestra_prior.wav
'''

from __future__ import annotations
from . import engine
from . import generate

import argparse
import pathlib
import praxis
import praxis.log as log
import time
import torch


def _add_train_args(sp: argparse.ArgumentParser) -> None:
    # datos y segmentación
    sp.add_argument('--audio', nargs='+', required=True, help='uno o más archivos de audio para entrenar')
    sp.add_argument('--out-dir', type=pathlib.Path, required=True, help='carpeta donde se guardan checkpoints y logs')
    sp.add_argument('--sample-rate', type=int, default=16_000, help='frecuencia de muestreo objetivo en Hz; los archivos ' 'se resamplean si no coinciden')
    sp.add_argument('--segment-length', type=int, default=32_768, help='muestras por segmento de entrenamiento; a 16 kHz ' '32768 son 2 segundos; debe ser múltiplo del ' 'total_stride del modelo (256 con los defaults)')
    sp.add_argument('--hop-length', type=int, default=None, help='paso entre segmentos consecutivos en muestras; ' 'controla el solape; por defecto segment_length / 2')
    sp.add_argument('--val-fraction', type=float, default=0.1, help='fracción de segmentos reservados para validación')
    sp.add_argument('--batch-size', type=int, default=8, help='segmentos por paso; escala con la VRAM disponible')
    sp.add_argument('--num-workers', type=int, default=2, help='procesos paralelos del DataLoader')

    # arquitectura del modelo
    sp.add_argument('--n-bands', type=int, default=16, help='número de subbandas del PQMF')
    sp.add_argument('--pqmf-taps', type=int, default=126, help='taps del prototipo PQMF, debe ser par; más taps da ' 'mejor filtro con más costo')
    sp.add_argument('--hidden-channels', type=int, default=64, help='canales del primer bloque del encoder; se duplican ' 'en cada bloque siguiente')
    sp.add_argument('--strides', type=int, nargs='+', default=[2, 4, 2], help='factores de diezmado por bloque del encoder; su ' 'producto multiplicado por n_bands da el ' 'total_stride')
    sp.add_argument('--latent-dim', type=int, default=64, help='dimensión del espacio latente')
    sp.add_argument('--n-res-per-block', type=int, default=3, help='unidades residuales por bloque')

    # pérdidas
    sp.add_argument('--fft-sizes', type=int, nargs='+', default=[512, 1024, 2048], help='escalas FFT para la pérdida STFT multi-escala')
    sp.add_argument('--stft-weight', type=float, default=1.0, help='peso del término espectral (convergencia + ' 'log-magnitud)')
    sp.add_argument('--waveform-weight', type=float, default=0.1, help='peso del L1 sobre la forma de onda')
    sp.add_argument('--beta-max', type=float, default=0.1, help='peso final del término KL tras el warmup')
    sp.add_argument('--warmup-steps', type=int, default=10_000, help='pasos del warmup lineal para el KL desde 0 hasta ' 'beta-max')

    # optimización
    sp.add_argument('--lr', type=float, default=1e-4, help='learning rate de Adam')
    sp.add_argument('--grad-clip', type=float, default=1.0, help='umbral para clipping de gradiente por norma')
    sp.add_argument('--n-steps', type=int, default=100_000, help='total de pasos de entrenamiento')

    # loop y checkpoints
    sp.add_argument('--log-every', type=int, default=100, help='pasos entre líneas de log en stdout y CSV')
    sp.add_argument('--val-every', type=int, default=1_000, help='pasos entre corridas de validación')
    sp.add_argument('--ckpt-every', type=int, default=5_000, help='pasos entre guardado de checkpoints')
    sp.add_argument('--keep-last-ckpts', type=int, default=3, help='cuántos checkpoints periódicos retener; los más ' 'antiguos se borran; rave_final.pt no se borra')
    sp.add_argument('--seed', type=int, default=42, help='semilla para reproducibilidad')
    sp.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='dispositivo de cómputo')


def _run_info(args: argparse.Namespace) -> None:
    generate.print_checkpoint_info(generate.describe_checkpoint(args.checkpoint))


def _run_reconstruct(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, config = generate.load_model(args.checkpoint, device)
    generate.reconstruct(
        model,
        args.audio,
        args.output,
        sample_rate=config['sample_rate'],
        device=device,
    )


def _run_sample(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model, config = generate.load_model(args.checkpoint, device)
    generate.sample_prior(model, args.output, duration_seconds=args.duration, sample_rate=config['sample_rate'], device=device, seed=args.seed)
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
        lr=args.lr,
        grad_clip=args.grad_clip,
        n_steps=args.n_steps,
        log_every=args.log_every,
        val_every=args.val_every,
        ckpt_every=args.ckpt_every,
        keep_last_ckpts=args.keep_last_ckpts,
        seed=args.seed,
        device=args.device,
    )
    engine.train(config)
    log.info(f'Duración del entrenamiento: {praxis.time.seconds_to_hms(time.perf_counter() - start_time)}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rave')
    sub = parser.add_subparsers(dest='command', required=True)

    train_p = sub.add_parser('train', help='entrenar el VAE tipo RAVE')
    _add_train_args(train_p)

    rec_p = sub.add_parser('reconstruct', help='reconstruir un archivo')
    rec_p.add_argument('--checkpoint', type=pathlib.Path, required=True, help='path al .pt a cargar')
    rec_p.add_argument('--audio', type=pathlib.Path, required=True, help='archivo de audio a reconstruir')
    rec_p.add_argument('--output', type=pathlib.Path, required=True, help='path del archivo de salida')
    rec_p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='dispositivo de cómputo')

    sam_p = sub.add_parser('sample', help='muestrear desde el prior')
    sam_p.add_argument('--checkpoint', type=pathlib.Path, required=True, help='path al .pt a cargar')
    sam_p.add_argument('--output', type=pathlib.Path, required=True, help='path del archivo de salida')
    sam_p.add_argument('--duration', type=float, required=True, help='duración en segundos del audio generado desde ' 'el prior')
    sam_p.add_argument('--seed', type=int, default=None, help='semilla para reproducibilidad; sin ella cada ' 'corrida genera algo distinto')
    sam_p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='dispositivo de cómputo')

    info_p = sub.add_parser('info', help='Inspeccionar metadatos de un checkpoint')
    info_p.add_argument('--checkpoint', type=pathlib.Path, required=True, help='Ruta al archivo .pt del checkpoint a inspeccionar')

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
    raise SystemExit(main())
