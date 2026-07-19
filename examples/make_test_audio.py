'''Genera un WAV de prueba (barrido + armónicos) para probar el pipeline sin
material propio. Uso: python examples/make_test_audio.py test.wav'''

import sys

import numpy as np
import soundfile as sf

sr = 16000
dur = 4.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)
f0 = 110.0 * 2 ** (2 * t / dur)  # barrido de 110 Hz a 2 octavas
x = np.sin(2 * np.pi * f0 * t) + 0.4 * np.sin(2 * np.pi * 2 * f0 * t) + 0.2 * np.sin(2 * np.pi * 3 * f0 * t)
x *= 0.3 * (0.6 + 0.4 * np.sin(2 * np.pi * 1.5 * t))  # envolvente lenta
x = x.astype(np.float32)

out = sys.argv[1] if len(sys.argv) > 1 else 'test.wav'
sf.write(out, x, sr)
print(f'escrito {out} ({dur:.0f}s @ {sr} Hz)')
