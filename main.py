'''Punto de entrada.

Ejemplos:
  # Modo archivo (headless, escribe WAV). --duration = segundos de AUDIO.
  python -m realtime_wavenet.main --input in.wav --mode file \
      --duration 10 --output out.wav

  # En GPU (acelera el entrenamiento; la generación puede no acelerar hasta
  # que se compile o se genere en lotes):
  python -m realtime_wavenet.main --input in.wav --mode file \
      --duration 10 --device cuda

  # Modo vivo (reproduce en tiempo real). Ctrl-C para detener.
  python -m realtime_wavenet.main --input in.wav --mode live
'''

import argparse
import core
import dataclasses
import pathlib
import praxis.log as log
import time


def build_args():
    parser = argparse.ArgumentParser(description='WaveNet online en tiempo real.')

    parser.add_argument('--input', required=True, help='WAV de entrada (se lee en bucle)')
    parser.add_argument('--mode', choices=['file', 'live', 'batched'], default='file')
    parser.add_argument('--output', type=pathlib.Path, default=pathlib.Path('out.wav'), help='salida en modo file')
    parser.add_argument('--out-dir', type=pathlib.Path, default=pathlib.Path('out_batched'), help='directorio de salida en modo batched')
    parser.add_argument('--step', type=float, default=5.0, help='modo batched: segundos por iteración (X entra, X sale)')
    parser.add_argument('--iterations', type=int, default=20, help='modo batched: número de iteraciones')
    parser.add_argument('--duration', type=float, default=20.0, help='modo file: segundos de AUDIO a generar')
    parser.add_argument('--max-wall', type=float, default=None, help='modo file: tope de segundos de reloj (evita cuelgues)')
    parser.add_argument('--temperature', type=float, default=None, help='sobrescribe la temperatura de muestreo')
    parser.add_argument('--realtime-pace', action='store_true', help='fuerza a la fuente de archivo a emular el reloj')
    parser.add_argument('--underrun', choices=['silence', 'hold'], default='silence')
    parser.add_argument('--device', default=None, help="'cpu', 'cuda', 'cuda:0'... (default del Config: cpu)")

    return parser.parse_args()


def main() -> int:
    args = build_args()
    cfg = core.Config()

    if args.temperature is not None:
        cfg = dataclasses.replace(cfg, temperature=args.temperature)

    if args.device is not None:
        cfg = dataclasses.replace(cfg, device=args.device)

    source = core.make_source('file', path=args.input, target_sr=cfg.sample_rate, realtime_pace=args.realtime_pace)

    if args.mode == 'batched':
        log.info(f'Modo batched | device: {cfg.device} | step: {args.step}s ' f'| iteraciones: {args.iterations} | out_dir: {args.out_dir}')

        def on_iter(i, loss, t_r, t_t, t_g, target, path):
            total = t_r + t_t + t_g
            factor = total / target
            log.info(f'iter {i:04d}  loss={loss:.4f}  ' f'read={t_r:.2f}s  train={t_t:.2f}s  gen={t_g:.2f}s  ' f'| total={total:.2f}s vs target={target:.1f}s ' f'({factor:.1f}x reloj)  -> {path}')

        start_time = time.perf_counter()
        core.run_batched(cfg, source, args.out_dir, args.step, args.iterations, on_iter=on_iter)
        log.info(f'Duración de run_batched: {time.perf_counter() - start_time:.6f} segundos')

        return 0

    if args.mode == 'file':
        sink = core.FileSink(args.output, cfg.sample_rate)
    else:
        sink = core.LiveSink(cfg.sample_rate, cfg.out_capacity, blocksize=cfg.blocksize, underrun=args.underrun)

    engine = core.Engine(cfg, source, sink)
    log.info(f'Modelo: {engine.train_model.param_count():,} parámetros ' f'| campo receptivo: {cfg.receptive_field} muestras ' f'({cfg.receptive_field / cfg.sample_rate * 1000:.0f} ms) ' f'| device: {cfg.device}')

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
                    log.info(f'Tope de reloj alcanzado ({wall_cap:.0f}s); ' f'corta antes de completar el audio pedido')
                    break

                done = sink.n_samples / target
                log.info(f'\taudio generado: {sink.n_samples / cfg.sample_rate:.2f}s ' f'/ {args.duration:.1f}s ({done * 100:.0f}%)  ' f'[reloj {time.time() - t0:.0f}s]')
                time.sleep(1.0)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        log.error('Deteniendo...')
    finally:
        engine.stop()

        if args.mode == 'file':
            log.info(f'salida escrita en {args.output} ' f'({sink.n_samples / cfg.sample_rate:.2f}s de audio)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
