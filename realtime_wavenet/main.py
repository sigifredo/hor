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
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out.wav'), help='salida en modo file')
    parser.add_argument('--duration', type=float, default=20.0, help='modo file: segundos de AUDIO a generar')
    parser.add_argument('--max-wall', type=float, default=None, help='modo file: tope de segundos de reloj (evita cuelgues)')
    parser.add_argument('--temperature', type=float, default=None, help='sobrescribe la temperatura de muestreo')
    parser.add_argument('--realtime-pace', action='store_true', help='fuerza a la fuente de archivo a emular el reloj')
    parser.add_argument('--underrun', choices=['silence', 'hold'], default='silence')

    return parser.parse_args()


def main() -> int:
    args = build_args()
    cfg = Config()

    if args.temperature is not None:
        from dataclasses import replace

        cfg = replace(cfg, temperature=args.temperature)

    source = make_source('file', path=args.input, target_sr=cfg.sample_rate, realtime_pace=args.realtime_pace)

    if args.mode == 'file':
        sink = FileSink(args.output, cfg.sample_rate)
    else:
        sink = LiveSink(cfg.sample_rate, cfg.out_capacity, blocksize=cfg.blocksize, underrun=args.underrun)

    engine = Engine(cfg, source, sink)
    log.info(f'Modelo: {engine.train_model.param_count():,} parámetros | campo receptivo: {cfg.receptive_field} muestras ({cfg.receptive_field / cfg.sample_rate * 1000:.0f} ms)')

    def on_round(i, loss):
        log.info(f'Ronda {i:04d}, loss={loss:.4f}')

    start_time = time.perf_counter()
    engine.start(on_round=on_round)
    log.info(f'Duración del entrenamiento: {time.perf_counter() - start_time:.6f} segundos')

    try:
        if args.mode == 'file':
            target = int(args.duration * cfg.sample_rate)
            wall_cap = args.max_wall if args.max_wall is not None else args.duration * 60
            t0 = time.time()

            while sink.n_samples < target:
                if time.time() - t0 > wall_cap:
                    log.info(f'Tope de reloj alcanzado ({wall_cap:.0f}s); corta antes de completar el audio pedido')
                    break

                done = sink.n_samples / target
                log.info(f'\taudio generado: {sink.n_samples / cfg.sample_rate:.2f}s / {args.duration:.1f}s ({done * 100:.0f}%)  [reloj {time.time() - t0:.0f}s]')
                time.sleep(1.0)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        log.error('Deteniendo...')
    finally:
        engine.stop()

        if args.mode == 'file':
            log.info(f'salida escrita en {args.output} ({sink.n_samples / cfg.sample_rate:.2f}s de audio)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
