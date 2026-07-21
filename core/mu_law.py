'''Compansión mu-law: float32 en [-1, 1] <-> índices enteros en [0, mu].

Implementación a mano (sin torchaudio) para control total y cero dependencia
extra. La salida del encoder son índices categóricos que consume tanto el
acumulador (objetivo de entrenamiento) como el muestreo en generación.'''

import numpy as np


def mu_law_encode(x: np.ndarray, mu: int = 255) -> np.ndarray:
    '''float32 en [-1, 1] -> int64 en [0, mu].'''
    x = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    magnitude = np.log1p(mu * np.abs(x)) / np.log1p(mu)
    signal = np.sign(x) * magnitude                  # [-1, 1] comprimido
    quantized = ((signal + 1.0) / 2.0 * mu + 0.5)    # [0, mu]
    return quantized.astype(np.int64)


def mu_law_decode(y: np.ndarray, mu: int = 255) -> np.ndarray:
    '''int en [0, mu] -> float32 en [-1, 1].'''
    y = np.asarray(y, dtype=np.float32)
    signal = 2.0 * (y / mu) - 1.0                    # [-1, 1] comprimido
    magnitude = (1.0 / mu) * ((1.0 + mu) ** np.abs(signal) - 1.0)
    return (np.sign(signal) * magnitude).astype(np.float32)


# Índice mu-law del silencio (0.0), útil para inicializar la generación.
SILENCE_INDEX = int(mu_law_encode(np.array([0.0], dtype=np.float32))[0])
