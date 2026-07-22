#!/usr/bin/env python3
'''Preentrenamiento offline.

Recorre el WAV en bucle dando pasos de Adam sobre ventanas de chunk_len y
guarda un checkpoint que main.py consume con --checkpoint. Ctrl-C guarda el
estado antes de salir; --resume continúa un checkpoint previo (mismo Adam,
mismos pesos), así que un preentrenamiento largo puede partirse en sesiones.

Ejemplos:
  python pretrain.py --input in.wav --steps 5000 --device cuda --out ckpt.pt
  python pretrain.py --input in.wav --steps 5000 --resume ckpt.pt --out ckpt.pt
  python pretrain.py --input in.wav --steps 0 --passes 4 --device cuda --out ckpt.pt
  python main.py --input in.wav --mode batched --checkpoint ckpt.pt ...
'''

import argparse
import core
import dataclasses
import math
import pathlib
import praxis.log as log
import soundfile as sf
import time
import torch
import tqdm

from core.checkpoint import load_checkpoint, save_checkpoint
from core.engine import train_step
from core.model import WaveNet
from core.mu_law import mu_law_encode


def build_args():
    parser = argparse.ArgumentParser(description='Preentrenamiento offline de la WaveNet.')

    parser.add_argument('--device', default=None, help='"cpu", "cuda", "cuda:0"... (default del Config: cpu)')
    parser.add_argument('--history', type=pathlib.Path, default=None, help='CSV de historial step,loss; por defecto --out con extensión .csv')
    parser.add_argument('--input', required=True, help='WAV de entrenamiento (se lee en bucle)')
    parser.add_argument('--log-every', type=int, default=50, help='pasos entre líneas de log')
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('checkpoint.pt'), help='ruta del checkpoint de salida')
    parser.add_argument('--passes', type=int, default=4, help='pasadas completas al archivo cuando --steps 0')
    parser.add_argument('--resume', type=pathlib.Path, default=None, help='checkpoint previo desde el que continuar')
    parser.add_argument('--steps', type=int, default=5000, help='pasos de Adam (1 paso = chunk_len muestras = 1 s de audio); 0 = automático: --passes pasadas completas al archivo')

    return parser.parse_args()


def auto_steps(input_path, cfg, passes: int):
    '''Pasos para recorrer el archivo completo `passes` veces.

    Un paso consume chunk_len muestras a cfg.sample_rate (1 s con el Config
    actual). La duración se toma de la cabecera del archivo (frames / sr
    nativo), que es invariante al remuestreo de la fuente.'''
    info = sf.info(str(input_path))
    duration = info.frames / info.samplerate
    per_pass = math.ceil(duration * cfg.sample_rate / cfg.chunk_len)
    return per_pass * passes, per_pass, duration


def main() -> int:
    args = build_args()
    cfg = core.Config()

    if args.device is not None:
        cfg = dataclasses.replace(cfg, device=args.device)

    steps = args.steps
    if steps == 0:
        steps, per_pass, duration = auto_steps(args.input, cfg, args.passes)
        log.info(f'steps automático: {steps} ({args.passes} pasadas x {per_pass} pasos/pasada; ' f'archivo de {duration / 60:.1f} min)')

    source = core.make_source('file', path=args.input, target_sr=cfg.sample_rate)
    model = WaveNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    step0 = 0
    if args.resume is not None:
        meta = load_checkpoint(args.resume, model, optimizer, cfg)
        step0 = int(meta.get('steps', 0))
        log.info(f'reanudando desde {args.resume} (pasos previos: {step0}, loss: {meta.get("loss", "?")})')

    log.info(f'Preentrenamiento | device: {cfg.device} | pasos: {steps} ' f'| {model.param_count():,} parámetros | ventana: {cfg.chunk_len / cfg.sample_rate:.1f}s | out: {args.out}')

    history_path = args.history if args.history is not None else args.out.with_suffix('.csv')
    log.info(f'Historial de loss: {history_path}')

    model.train()
    step = step0
    loss_val = float('nan')
    reloj_str = '?'
    last_t = time.perf_counter()
    history = []

    try:
        bar = tqdm.tqdm(range(step0 + 1, step0 + steps + 1), desc='preentrenamiento', unit='paso')

        for step in bar:
            chunk_mu = mu_law_encode(source.read(cfg.chunk_len), cfg.mu)
            x = torch.from_numpy(chunk_mu).long().unsqueeze(0).to(cfg.device)
            # .item() sincroniza CUDA: la pérdida es real y el ritmo medido, honesto.
            loss_val = float(train_step(model, optimizer, x, cfg).item())

            if step % args.log_every == 0:
                now = time.perf_counter()
                audio_s = args.log_every * cfg.chunk_len / cfg.sample_rate
                reloj_str = f'{audio_s / (now - last_t):.1f}x'
                last_t = now

            history.append((step, loss_val))
            bar.set_postfix(loss=f'{loss_val:.4f}', reloj=reloj_str)
    except KeyboardInterrupt:
        log.error('Interrumpido; guardando checkpoint...')

    with open(history_path, 'w') as hist_f:
        hist_f.write('step,loss\n')

        for it in history:
            hist_f.write(f'{it[0]},{it[1]}\n')

    save_checkpoint(args.out, model, optimizer, cfg, meta={'steps': step, 'loss': loss_val, 'input': str(args.input)})
    log.info(f'checkpoint guardado en {args.out} (pasos: {step}, loss: {loss_val:.4f})')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
