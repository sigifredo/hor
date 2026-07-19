'''Puerto de salida (simétrico a sources) + adaptadores.

FileSink acumula en memoria y escribe un WAV al cerrar: permite ejecutar y
probar el sistema en máquinas sin dispositivo de audio (headless).

LiveSink reproduce en tiempo real vía sounddevice. Un ring buffer desacopla la
generación del callback de audio: write() bloquea cuando el ring está lleno
(contrapresión que pacea la generación al ritmo de reproducción = tiempo real);
el callback nunca bloquea a la espera de datos y ante déficit rellena según el
modo de underrun elegido.

Nota de honestidad técnica: el callback adquiere brevemente un lock (Condition).
Bajo el GIL, una ruta de audio verdaderamente lock-free en Python no es
alcanzable; esta es la versión experimental v1 acordada (threading + buffers
generosos). El escape estructural, si la contención resulta audible, es
multiprocessing.'''

import threading
from abc import ABC, abstractmethod

import numpy as np
import soundfile as sf


class AudioSink(ABC):
    @abstractmethod
    def write(self, samples: np.ndarray) -> None:
        ...

    def close(self) -> None:
        pass


class FileSink(AudioSink):
    def __init__(self, path: str, sample_rate: int):
        self.path = path
        self.sr = sample_rate
        self._chunks = []

    def write(self, samples: np.ndarray) -> None:
        self._chunks.append(np.asarray(samples, dtype=np.float32))

    def close(self) -> None:
        if self._chunks:
            sf.write(self.path, np.concatenate(self._chunks), self.sr)


class LiveSink(AudioSink):
    def __init__(self, sample_rate: int, capacity: int,
                 blocksize: int = 1024, underrun: str = 'silence'):
        import sounddevice as sd
        self.sr = sample_rate
        self.cap = capacity
        self.buf = np.zeros(capacity, dtype=np.float32)
        self.rpos = 0
        self.wpos = 0
        self.count = 0
        self.cond = threading.Condition()
        self.underrun = underrun          # 'silence' o 'hold'
        self.last = 0.0
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1,
            blocksize=blocksize, callback=self._callback)
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        with self.cond:
            n = min(frames, self.count)
            if n > 0:
                first = min(n, self.cap - self.rpos)
                outdata[:first, 0] = self.buf[self.rpos:self.rpos + first]
                if first < n:
                    outdata[first:n, 0] = self.buf[:n - first]
                self.rpos = (self.rpos + n) % self.cap
                self.count -= n
                self.last = float(outdata[n - 1, 0])
            if n < frames:                # underrun
                fill = 0.0 if self.underrun == 'silence' else self.last
                outdata[n:, 0] = fill
            self.cond.notify()

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32)
        i = 0
        with self.cond:
            while i < len(samples):
                while self.count >= self.cap:
                    self.cond.wait()
                space = self.cap - self.count
                k = min(space, len(samples) - i)
                first = min(k, self.cap - self.wpos)
                self.buf[self.wpos:self.wpos + first] = samples[i:i + first]
                if first < k:
                    self.buf[:k - first] = samples[i + first:i + k]
                self.wpos = (self.wpos + k) % self.cap
                self.count += k
                i += k
                self.cond.notify()

    def close(self) -> None:
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass
