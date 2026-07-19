'''Puerto de entrada (arquitectura hexagonal) + adaptadores.

Patrón: Strategy a nivel de comportamiento, puerto-adaptador a nivel
estructural. El core depende solo de la abstracción AudioSource; cada fuente
concreta encapsula su propia semántica de reloj (pull inmediato para archivo,
push bloqueante para micrófono en vivo en el futuro).

Contrato del puerto: read(n) devuelve exactamente n muestras float32 mono en
[-1, 1] a TARGET_SR. Todo resampleo y downmix ocurre dentro del adaptador.'''

from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


class AudioSource(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        ...

    @abstractmethod
    def read(self, n: int) -> np.ndarray:
        '''Devuelve exactamente n muestras float32 mono. Bloquea en fuentes en vivo.'''
        ...

    def close(self) -> None:
        pass


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32)
    try:
        import soxr
        return soxr.resample(x, sr_in, sr_out).astype(np.float32)
    except ImportError:
        # Fallback sin dependencia extra (calidad menor).
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class FileLoopSource(AudioSource):
    '''Lee un archivo a memoria y lo recorre en bucle infinito.

    El indexado modular reproduce el bucle sin ramas y tila automáticamente si n
    excede la longitud del archivo. Con realtime_pace=True el adaptador duerme
    para emular el reloj de 16 kHz (1 s de archivo = 1 s de reloj); por defecto
    consume a máxima velocidad, lo que expone antes el comportamiento del
    sistema en la fase experimental.'''

    def __init__(self, path: str, target_sr: int = 16000,
                 realtime_pace: bool = False):
        data, sr = sf.read(path, dtype='float32', always_2d=True)
        data = data.mean(axis=1)                     # downmix a mono
        data = _resample(data, sr, target_sr)
        self._buf = np.ascontiguousarray(data, dtype=np.float32)
        if len(self._buf) == 0:
            raise ValueError('archivo de audio vacío')
        self._sr = target_sr
        self._pos = 0
        self._pace = realtime_pace

    @property
    def sample_rate(self) -> int:
        return self._sr

    def read(self, n: int) -> np.ndarray:
        L = len(self._buf)
        idx = (self._pos + np.arange(n)) % L
        out = self._buf[idx]
        self._pos = (self._pos + n) % L
        if self._pace:
            import time
            time.sleep(n / self._sr)
        return out


# --- Registro / fábrica ---------------------------------------------------
# Agregar una fuente futura (p. ej. 'mic', 'sensor') es una línea aquí; el core
# (acumulador, entrenamiento, generación) no cambia.
_SOURCES = {
    'file': FileLoopSource,
}


def make_source(kind: str, **kwargs) -> AudioSource:
    if kind not in _SOURCES:
        raise KeyError(f'fuente desconocida: {kind!r}. '
                       f'Disponibles: {list(_SOURCES)}')
    return _SOURCES[kind](**kwargs)
