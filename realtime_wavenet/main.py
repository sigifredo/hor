'''Punto de entrada.

Ejemplos:
  # Modo archivo (headless, escribe WAV). Ejecuta N segundos y guarda salida.
  python -m realtime_wavenet.main --input in.wav --mode file \
      --duration 20 --output out.wav

  # Modo vivo (reproduce en tiempo real). Ctrl-C para detener.
  python -m realtime_wavenet.main --input in.wav --mode live
'''

import argparse
import pathlib
import praxis.log as log
import time

from .config import Config
from .engine import Engine
from .sources import make_source
from .sinks import FileSink, LiveSink


def build_args():
    parser = argparse.ArgumentParser(description='WaveNet online en tiempo real.')

    parser.add_argument('--input', required=True, help='WAV de entrada (se lee en bucle)')
    parser.add_argument('--mode', choices=['file', 'live'], default='file')
    parser.add_argument('--output', type=pathlib.Path, default='out.wav', help='salida en modo file')
    parser.add_argument('--duration', type=float, default=20.0, help='segundos de ejecución en modo file')
    parser.add_argument('--temperature', type=float, default=None, help='sobrescribe la temperatura de muestreo')
    parser.add_argument('--realtime-pace', action='store_true', help='fuerza a la fuente de archivo a emular el reloj')
    parser.add_argument('--underrun', choices=['silence', 'hold'], default='silence')

    return parser.parse_args()


def main():
    a = build_args()
    cfg = Config()

    if a.temperature is not None:
        from dataclasses import replace

        cfg = replace(cfg, temperature=a.temperature)

    source = make_source('file', path=a.input, target_sr=cfg.sample_rate, realtime_pace=a.realtime_pace)

    if a.mode == 'file':
        sink = FileSink(a.output, cfg.sample_rate)
    else:
        sink = LiveSink(cfg.sample_rate, cfg.out_capacity, blocksize=cfg.blocksize, underrun=a.underrun)

    engine = Engine(cfg, source, sink)
    log.info(f'Modelo: {engine.train_model.param_count():,} parámetros | campo receptivo: {cfg.receptive_field} muestras ({cfg.receptive_field / cfg.sample_rate * 1000:.0f} ms)')

    def on_round(i, loss):
        log.info(f'Ronda {i:04d}, loss={loss:.4f}')

    engine.start(on_round=on_round)

    try:
        if a.mode == 'file':
            time.sleep(a.duration)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        log.error('Deteniendo...')
    finally:
        engine.stop()

        if a.mode == 'file':
            log.info(f'Salida escrita en {a.output}')


if __name__ == '__main__':
    main()
