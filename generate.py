#!/usr/bin/env python3
'''Generación independiente desde un checkpoint, sin entrenamiento.

Carga los pesos (pretrain.py o los guardados por main.py durante el
entrenamiento), genera la duración pedida con el generador incremental
NumPy y escribe un WAV. No necesita GPU: la generación vive en CPU por
diseño. Ctrl-C escribe el audio generado hasta el momento.

Ejemplos:
  python generate.py --checkpoint ckpt.pt --duration 30 --out gen.wav
  python generate.py --checkpoint ckpt.pt --duration 30 --temperature 0.85
'''

import argparse
import core
import numpy as np
import pathlib
import praxis.log as log
import soundfile as sf
import time

from core.checkpoint import load_checkpoint
from core.model import WaveNet, WaveNetGenerator, cpu_state_dict
from core.mu_law import mu_law_decode


def build_args():
    parser = argparse.ArgumentParser(description='Generación autoregresiva desde un checkpoint.')

    parser.add_argument('--checkpoint', type=pathlib.Path, required=True, help='checkpoint de pretrain.py o de main.py')
    parser.add_argument('--duration', type=float, default=30.0, help='segundos de audio a generar')
    parser.add_argument('--out', type=pathlib.Path, default=pathlib.Path('gen.wav'), help='WAV de salida')
    parser.add_argument('--temperature', type=float, default=None, help='sobrescribe la temperatura de muestreo del Config')

    return parser.parse_args()


def main() -> int:
    args = build_args()
    cfg = core.Config()
    temperature = args.temperature if args.temperature is not None else cfg.temperature

    model = WaveNet(cfg)  # CPU: la generación no usa GPU
    meta = load_checkpoint(args.checkpoint, model, cfg=cfg)
    log.info(f'checkpoint {args.checkpoint} | pasos: {meta.get("steps", "?")} ' f'| loss: {meta.get("loss", "?")} | temperatura: {temperature}')

    generator = WaveNetGenerator(cfg)
    generator.load_state(cpu_state_dict(model), 0)

    block = cfg.sample_rate  # 1 s por bloque, para poder reportar avance
    total = int(args.duration * cfg.sample_rate)
    chunks = []
    done = 0
    t0 = time.perf_counter()

    try:
        while done < total:
            n = min(block, total - done)
            chunks.append(mu_law_decode(generator.generate(n, temperature), cfg.mu))
            done += n

            if done % (5 * cfg.sample_rate) == 0 or done == total:
                dt = time.perf_counter() - t0
                log.info(f'generado {done / cfg.sample_rate:.0f}s / {args.duration:.0f}s ' f'({done / cfg.sample_rate / dt:.2f}x reloj)')
    except KeyboardInterrupt:
        log.error('Interrumpido; escribiendo el audio generado hasta ahora...')

    if not chunks:
        log.error('nada que escribir')
        return 1

    audio = np.concatenate(chunks)
    sf.write(str(args.out), audio, cfg.sample_rate)
    log.info(f'Salida escrita en {args.out} ({len(audio) / cfg.sample_rate:.2f}s de audio)')
    log.info(f'Tiempo de entrenamiento: {time.perf_counter() - t0}s')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
