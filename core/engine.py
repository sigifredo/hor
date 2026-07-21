'''Orquestación concurrente.

Unidades de ejecución:
  - Acumulador: consume la fuente, codifica mu-law, arma chunks, los encola.
  - Entrenamiento: consume chunks, forward+backward, paso de Adam, publica
    pesos vía swap atómico. train_model es persistente (warm start + estado de
    Adam conservado entre rondas).
  - Generación: lee el puntero de pesos por bloque, produce muestras con caché
    incremental, las escribe al sink.

Protocolo de pesos: dos instancias. train_model nunca lo lee generación; tras
cada ronda se clona una copia congelada fresca y se reasigna el puntero. La
generación ve pesos viejos o nuevos, nunca un estado a medias.

Contrapresión: la cola de chunks es acotada (queue_maxsize). Con archivo, el
pacing lo impone el throughput de entrenamiento, no el reloj de pared.'''

from .config import Config
from .model import WaveNet, WaveNetGenerator, clone_for_inference
from .mu_law import mu_law_encode, mu_law_decode

import numpy as np
import queue
import threading
import torch
import torch.nn.functional as F


class SharedWeights:
    '''Puntero de pesos para generación. set() en el hilo de entrenamiento,
    get() en el de generación.'''

    def __init__(self, model: WaveNet):
        self._lock = threading.Lock()
        self._model = model

    def get(self) -> WaveNet:
        with self._lock:
            return self._model

    def set(self, model: WaveNet) -> None:
        with self._lock:
            self._model = model


def accumulator_loop(source, chunk_queue, stop_event, cfg: Config):
    while not stop_event.is_set():
        chunk_f = source.read(cfg.chunk_len)
        chunk_mu = mu_law_encode(chunk_f, cfg.mu)
        while not stop_event.is_set():
            try:
                chunk_queue.put(chunk_mu, timeout=0.2)
                break
            except queue.Full:
                continue


def training_loop(chunk_queue, train_model, shared, stop_event, cfg: Config, on_round=None):
    optimizer = torch.optim.Adam(train_model.parameters(), lr=cfg.lr)
    round_idx = 0
    while not stop_event.is_set():
        try:
            chunk = chunk_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        dev = next(train_model.parameters()).device
        x = torch.from_numpy(chunk).long().unsqueeze(0).to(dev)  # (1, T)
        logits = train_model(x)  # (1, Q, T)
        pred = logits[:, :, :-1]  # predice t+1 desde t
        target = x[:, 1:]
        w = cfg.loss_warmup
        pred = pred[:, :, w:]
        target = target[:, w:]
        loss = F.cross_entropy(pred.reshape(-1, cfg.quant_levels), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        shared.set(clone_for_inference(train_model))  # swap atómico
        round_idx += 1
        if on_round is not None:
            on_round(round_idx, float(loss.item()))


def generation_loop(shared, generator, sink, stop_event, cfg: Config):
    while not stop_event.is_set():
        generator.model = shared.get()  # relectura; cachés persisten
        block = np.empty(cfg.gen_block, dtype=np.int64)
        for i in range(cfg.gen_block):
            block[i] = generator.step(cfg.temperature)
        audio = mu_law_decode(block, cfg.mu)
        sink.write(audio)


class Engine:
    '''Arma y controla los tres hilos sobre una fuente y un sink dados.'''

    def __init__(self, cfg: Config, source, sink):
        self.cfg = cfg
        self.source = source
        self.sink = sink
        self.train_model = WaveNet(cfg).to(cfg.device)
        # clone_for_inference usa deepcopy, que preserva el device del original.
        self.shared = SharedWeights(clone_for_inference(self.train_model))
        self.generator = WaveNetGenerator(self.shared.get())
        self.chunk_queue = queue.Queue(maxsize=cfg.queue_maxsize)
        self.stop_event = threading.Event()
        self._threads = []

    def start(self, on_round=None):
        specs = [
            (accumulator_loop, (self.source, self.chunk_queue, self.stop_event, self.cfg)),
            (training_loop, (self.chunk_queue, self.train_model, self.shared, self.stop_event, self.cfg, on_round)),
            (generation_loop, (self.shared, self.generator, self.sink, self.stop_event, self.cfg)),
        ]
        for fn, args in specs:
            t = threading.Thread(target=fn, args=args, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self.stop_event.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self.source.close()
        self.sink.close()


def run_batched(cfg: Config, source, out_dir, step_seconds: float, iterations: int, on_iter=None):
    '''Modo secuencial con paridad temporal.

    Cada iteración: lee step_seconds del source, entrena una ronda con ese
    chunk, genera step_seconds, guarda 'out_XXXX.wav'. Sin hilos, sin swap:
    train y gen usan el mismo modelo, secuenciales. Ideal para medir el gap
    respecto al reloj de pared y para escuchar la evolución iteración a
    iteración.

    Las cachés del generador se conservan entre iteraciones (continuidad
    sonora + pequeño transitorio audible en cada nuevo bloque de pesos).'''
    import time
    from pathlib import Path
    import soundfile as sf

    from .mu_law import mu_law_encode, mu_law_decode

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(step_seconds * cfg.sample_rate)
    model = WaveNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    generator = WaveNetGenerator(model)

    for i in range(iterations):
        # 1. Leer step_seconds del source (simula "escuchar durante X s").
        t0 = time.perf_counter()
        chunk_f = source.read(n)
        chunk_mu = mu_law_encode(chunk_f, cfg.mu)
        t_read = time.perf_counter() - t0

        # 2. Entrenar una ronda con ese chunk.
        t0 = time.perf_counter()
        model.train()
        x = torch.from_numpy(chunk_mu).long().unsqueeze(0).to(cfg.device)
        logits = model(x)
        w = cfg.loss_warmup
        pred = logits[:, :, :-1][:, :, w:]
        target = x[:, 1:][:, w:]
        loss = F.cross_entropy(pred.reshape(-1, cfg.quant_levels), target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t_train = time.perf_counter() - t0

        # 3. Generar step_seconds.
        t0 = time.perf_counter()
        model.eval()
        block = np.empty(n, dtype=np.int64)
        for j in range(n):
            block[j] = generator.step(cfg.temperature)
        audio_out = mu_law_decode(block, cfg.mu)
        t_gen = time.perf_counter() - t0

        # 4. Guardar.
        out_path = out_dir / f'out_{i:04d}.wav'
        sf.write(str(out_path), audio_out, cfg.sample_rate)

        if on_iter is not None:
            on_iter(i, float(loss.item()), t_read, t_train, t_gen, step_seconds, str(out_path))
