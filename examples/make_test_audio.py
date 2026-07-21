'''Genera un WAV de prueba (barrido + armónicos) para probar el pipeline sin
material propio. Uso: python examples/make_test_audio.py test.wav'''

import argparse
import numpy as np
import pathlib
import praxis.log as log
import soundfile as sf


def main() -> int:
    parser = argparse.ArgumentParser(description='WaveNet online en tiempo real.')
    parser.add_argument('out_file', type=pathlib.Path, help='Archivo de salida')
    args = parser.parse_args()

    if args.out_file.is_file():
        log.error(f'El archivo de salida es inválido: {args.out_file}')
        return 1

    sr = 16_000
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    f0 = 110.0 * 2 ** (2 * t / dur)  # barrido de 110 Hz a 2 octavas
    x = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
    x *= 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))  # envolvente lenta
    x = x.astype(np.float32)

    sf.write(args.out_file, x, sr)
    log.info(f'Escrito {args.out_file} ({dur:.0f}s @ {sr} Hz)')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
