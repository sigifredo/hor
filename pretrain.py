#!/usr/bin/env python3
'''Preentrenamiento offline.

Recorre el WAV en bucle dando pasos de Adam sobre ventanas de chunk_len y
guarda un checkpoint que main.py consume con --checkpoint. Ctrl-C guarda el
estado antes de salir; --resume continúa un checkpoint previo (mismo Adam,
mismos pesos), así que un preentrenamiento largo puede partirse en sesiones.

Ejemplos:
  python pretrain.py --input in.wav --steps 5000 --device cuda --out ckpt.pt
  python pretrain.py --input in.wav --steps 5000 --resume ckpt.pt --out ckpt.pt
  python main.py --input in.wav --mode batched --checkpoint ckpt.pt ...
'''

import argparse
import core
import dataclasses
import pathlib
import praxis.log as log
import time
import torch

from core.checkpoint import load_checkpoint, save_checkpoint
from core.engine import train_step
from core.model import WaveNet
from core.mu_law import mu_law_encode


def build_args():
    parser = argparse.ArgumentParser(description='Preentrenamiento offline de la WaveNet.')

    parser.add_argument('--device', default=None, help='"cpu", "cuda", "cuda:0"... (default del Config: cpu)')
    parser.add_argument('--input', required=True, help='WAV de entrenamiento (se lee en bucle)')
    parser.add_argument('--log-every', type=int, default=50, help='pasos entre líneas de log')
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('checkpoint.pt'), help='ruta del checkpoint de salida')
    parser.add_argument('--resume', type=pathlib.Path, default=None, help='checkpoint previo desde el que continuar')
    parser.add_argument('--steps', type=int, default=5000, help='pasos de Adam (1 paso = chunk_len muestras = 1 s de audio)')

    return parser.parse_args()


def main() -> int:
    args = build_args()
    cfg = core.Config()

    if args.device is not None:
        cfg = dataclasses.replace(cfg, device=args.device)

    source = core.make_source('file', path=args.input, target_sr=cfg.sample_rate)
    model = WaveNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    step0 = 0
    if args.resume is not None:
        meta = load_checkpoint(args.resume, model, optimizer, cfg)
        step0 = int(meta.get('steps', 0))
        log.info(f'reanudando desde {args.resume} (pasos previos: {step0}, loss: {meta.get("loss", "?")})')

    log.info(f'Preentrenamiento | device: {cfg.device} | pasos: {args.steps} ' f'| {model.param_count():,} parámetros | ventana: {cfg.chunk_len / cfg.sample_rate:.1f}s | out: {args.out}')

    model.train()
    step = step0
    loss_val = float('nan')
    last_t = time.perf_counter()

    try:
        for step in range(step0 + 1, step0 + args.steps + 1):
            chunk_mu = mu_law_encode(source.read(cfg.chunk_len), cfg.mu)
            x = torch.from_numpy(chunk_mu).long().unsqueeze(0).to(cfg.device)
            # .item() sincroniza CUDA: la pérdida es real y el ritmo medido, honesto.
            loss_val = float(train_step(model, optimizer, x, cfg).item())

            if step % args.log_every == 0:
                now = time.perf_counter()
                audio_s = args.log_every * cfg.chunk_len / cfg.sample_rate
                log.info(f'paso {step:05d}  loss={loss_val:.4f}  ' f'{audio_s / (now - last_t):.1f}x audio/reloj')
                last_t = now
    except KeyboardInterrupt:
        log.error('Interrumpido; guardando checkpoint...')

    save_checkpoint(args.out, model, optimizer, cfg, meta={'steps': step, 'loss': loss_val, 'input': str(args.input)})
    log.info(f'checkpoint guardado en {args.out} (pasos: {step}, loss: {loss_val:.4f})')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
