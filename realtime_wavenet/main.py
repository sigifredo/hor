'''Punto de entrada.

Ejemplos:
  # Modo archivo (headless, escribe WAV). Ejecuta N segundos y guarda salida.
  python -m realtime_wavenet.main --input in.wav --mode file \
      --duration 20 --output out.wav

  # Modo vivo (reproduce en tiempo real). Ctrl-C para detener.
  python -m realtime_wavenet.main --input in.wav --mode live
'''

import argparse
import time

from .config import Config
from .engine import Engine
from .sources import make_source
from .sinks import FileSink, LiveSink


def build_args():
    p = argparse.ArgumentParser(description='WaveNet online en tiempo real.')
    p.add_argument('--input', required=True, help='WAV de entrada (se lee en bucle)')
    p.add_argument('--mode', choices=['file', 'live'], default='file')
    p.add_argument('--output', default='out.wav', help='salida en modo file')
    p.add_argument('--duration', type=float, default=20.0,
                   help='segundos de ejecución en modo file')
    p.add_argument('--temperature', type=float, default=None,
                   help='sobrescribe la temperatura de muestreo')
    p.add_argument('--realtime-pace', action='store_true',
                   help='fuerza a la fuente de archivo a emular el reloj')
    p.add_argument('--underrun', choices=['silence', 'hold'], default='silence')
    return p.parse_args()


def main():
    a = build_args()
    cfg = Config()
    if a.temperature is not None:
        from dataclasses import replace
        cfg = replace(cfg, temperature=a.temperature)

    source = make_source('file', path=a.input,
                         target_sr=cfg.sample_rate,
                         realtime_pace=a.realtime_pace)

    if a.mode == 'file':
        sink = FileSink(a.output, cfg.sample_rate)
    else:
        sink = LiveSink(cfg.sample_rate, cfg.out_capacity,
                        blocksize=cfg.blocksize, underrun=a.underrun)

    engine = Engine(cfg, source, sink)
    print(f'modelo: {engine.train_model.param_count():,} parámetros '
          f'| campo receptivo: {cfg.receptive_field} muestras '
          f'({cfg.receptive_field / cfg.sample_rate * 1000:.0f} ms)')

    def on_round(i, loss):
        print(f'ronda {i:04d}  loss={loss:.4f}')

    engine.start(on_round=on_round)
    try:
        if a.mode == 'file':
            time.sleep(a.duration)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        print('\ndeteniendo...')
    finally:
        engine.stop()
        if a.mode == 'file':
            print(f'salida escrita en {a.output}')


if __name__ == '__main__':
    main()
