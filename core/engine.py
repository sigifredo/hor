'''Orquestación concurrente.

Unidades de ejecución:
  - Acumulador: consume la fuente, codifica mu-law, arma chunks, los encola.
  - Entrenamiento: consume chunks, forward+backward, paso de Adam, publica
    una instantánea CPU de los pesos (versionada). train_model es persistente
    (warm start + estado de Adam conservado entre rondas).
  - Generación: corre siempre en CPU/NumPy. Recarga pesos solo cuando la
    versión publicada cambia, genera por bloques con caché incremental y
    escribe al sink.

Protocolo de pesos: SharedWeights guarda (versión, state_dict en CPU). El
entrenamiento hace set() tras cada paso; la generación compara versiones y
hace load_state() solo ante cambio. La generación ve pesos viejos o nuevos,
nunca un estado a medias.

Contrapresión: la cola de chunks es acotada (queue_maxsize). Con archivo, el
pacing lo impone el throughput de entrenamiento, no el reloj de pared.'''

from .checkpoint import load_checkpoint, save_checkpoint
from .config import Config
from .model import WaveNet, WaveNetGenerator, cpu_state_dict
from .mu_law import mu_law_encode, mu_law_decode

import queue
import threading
import torch
import torch.nn.functional as F


def _cuda_sync(device: str) -> None:
    '''Los kernels CUDA son asíncronos: sin sincronizar, perf_counter mide el
    encolado del trabajo, no su ejecución, y el costo real se filtra al
    siguiente tramo medido.'''
    if str(device).startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()


class SharedWeights:
    '''Instantánea versionada de pesos en CPU. set() en el hilo de
    entrenamiento, get() en el de generación.'''

    def __init__(self, state: dict):
        self._lock = threading.Lock()
        self._state = state
        self._version = 0

    def get(self):
        with self._lock:
            return self._version, self._state

    def set(self, state: dict) -> None:
        with self._lock:
            self._state = state
            self._version += 1


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


def train_step(model, optimizer, x, cfg: Config):
    '''Un paso de Adam sobre una ventana (1, T). Devuelve la pérdida.
    Compartido por training_loop, run_batched y pretrain.py.'''
    logits = model(x)  # (1, Q, T)
    w = cfg.loss_warmup
    pred = logits[:, :, :-1][:, :, w:]  # predice t+1 desde t
    target = x[:, 1:][:, w:]
    # F.cross_entropy acepta (B, C, T) con target (B, T) directamente.
    # NUNCA reshape(-1, C) sobre (B, C, T): reinterpreta la memoria mezclando
    # clases y tiempo, y produce un objetivo sin sentido que aun así "baja".
    loss = F.cross_entropy(pred, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return loss


def training_loop(chunk_queue, train_model, optimizer, shared, stop_event, cfg: Config, on_round=None, save_path=None, save_every=0, step0=0):
    round_idx = 0
    loss_val = float('nan')

    def save():
        save_checkpoint(save_path, train_model, optimizer, cfg, meta={'steps': step0 + round_idx, 'loss': loss_val})

    while not stop_event.is_set():
        try:
            chunk = chunk_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        dev = next(train_model.parameters()).device
        x = torch.from_numpy(chunk).long().unsqueeze(0).to(dev)  # (1, T)
        loss = train_step(train_model, optimizer, x, cfg)

        shared.set(cpu_state_dict(train_model))  # publicación versionada
        round_idx += 1
        loss_val = float(loss.item())

        if save_path is not None and save_every > 0 and round_idx % save_every == 0:
            save()

        if on_round is not None:
            on_round(round_idx, loss_val)

    if save_path is not None and round_idx > 0:
        save()  # estado final al detener


def generation_loop(shared, generator, sink, stop_event, cfg: Config):
    while not stop_event.is_set():
        version, state = shared.get()
        if version != generator.version:
            generator.load_state(state, version)  # cachés persisten
        block = generator.generate(cfg.gen_block, cfg.temperature)
        sink.write(mu_law_decode(block, cfg.mu))


class Engine:
    '''Arma y controla los tres hilos sobre una fuente y un sink dados.'''

    def __init__(self, cfg: Config, source, sink, checkpoint=None, save_path=None, save_every=0):
        self.cfg = cfg
        self.source = source
        self.sink = sink
        self.save_path = save_path
        self.save_every = save_every
        self.train_model = WaveNet(cfg).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.train_model.parameters(), lr=cfg.lr)

        self._step0 = 0
        if checkpoint is not None:
            meta = load_checkpoint(checkpoint, self.train_model, self.optimizer, cfg)
            self._step0 = int(meta.get('steps', 0))

        # La instantánea inicial se toma DESPUÉS de cargar el checkpoint: la
        # generación arranca desde los pesos preentrenados, no desde ruido.
        self.shared = SharedWeights(cpu_state_dict(self.train_model))
        self.generator = WaveNetGenerator(cfg)
        version, state = self.shared.get()
        self.generator.load_state(state, version)
        self.chunk_queue = queue.Queue(maxsize=cfg.queue_maxsize)
        self.stop_event = threading.Event()
        self._threads = []

    def start(self, on_round=None):
        specs = [
            (accumulator_loop, (self.source, self.chunk_queue, self.stop_event, self.cfg)),
            (training_loop, (self.chunk_queue, self.train_model, self.optimizer, self.shared, self.stop_event, self.cfg, on_round, self.save_path, self.save_every, self._step0)),
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


def run_batched(cfg: Config, source, out_dir, step_seconds: float, iterations: int, on_iter=None, checkpoint=None, save_path=None, save_every=0):
    '''Modo secuencial con paridad temporal.

    Cada iteración: lee step_seconds del source, entrena UNA RONDA POR VENTANA
    de chunk_len dentro de ese chunk (con step=5 s y chunk_len=1 s son 5 pasos
    de Adam por iteración), genera step_seconds con el generador NumPy, guarda
    'out_XXXX.wav'. La pérdida reportada es el promedio de las ventanas. Con
    `checkpoint` parte de un preentrenamiento (pretrain.py) en vez de cero.

    Con `save_path`, guarda el estado cada `save_every` iteraciones y siempre
    al terminar o interrumpir (finally), acumulando los pasos del checkpoint
    de entrada. Nunca escribe sobre `checkpoint`: son rutas independientes.

    Las cachés del generador se conservan entre iteraciones (continuidad
    sonora + pequeño transitorio audible en cada publicación de pesos).'''
    import time
    from pathlib import Path
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(step_seconds * cfg.sample_rate)
    model = WaveNet(cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    steps_done = 0
    mean_loss = float('nan')
    if checkpoint is not None:
        meta = load_checkpoint(checkpoint, model, optimizer, cfg)
        steps_done = int(meta.get('steps', 0))

    generator = WaveNetGenerator(cfg)

    try:
        for i in range(iterations):
            # 1. Leer step_seconds del source (simula "escuchar durante X s").
            t0 = time.perf_counter()
            chunk_f = source.read(n)
            chunk_mu = mu_law_encode(chunk_f, cfg.mu)
            t_read = time.perf_counter() - t0

            # 2. Entrenar: un paso de Adam por ventana de chunk_len.
            t0 = time.perf_counter()
            model.train()
            x_all = torch.from_numpy(chunk_mu).long().to(cfg.device)
            losses = []
            for s in range(0, n, cfg.chunk_len):
                xw = x_all[s : s + cfg.chunk_len].unsqueeze(0)
                if xw.shape[1] <= cfg.loss_warmup + 1:
                    break  # cola sin contexto suficiente
                loss = train_step(model, optimizer, xw, cfg)
                losses.append(float(loss.item()))
            _cuda_sync(cfg.device)
            t_train = time.perf_counter() - t0

            # 3. Publicar pesos al generador y generar step_seconds (CPU/NumPy).
            t0 = time.perf_counter()
            model.eval()
            generator.load_state(cpu_state_dict(model))
            block = generator.generate(n, cfg.temperature)
            audio_out = mu_law_decode(block, cfg.mu)
            t_gen = time.perf_counter() - t0

            # 4. Guardar.
            out_path = out_dir / f'out_{i:04d}.wav'
            sf.write(str(out_path), audio_out, cfg.sample_rate)

            steps_done += len(losses)
            mean_loss = sum(losses) / max(len(losses), 1)

            if save_path is not None and save_every > 0 and (i + 1) % save_every == 0:
                save_checkpoint(save_path, model, optimizer, cfg, meta={'steps': steps_done, 'loss': mean_loss})

            if on_iter is not None:
                on_iter(i, mean_loss, t_read, t_train, t_gen, step_seconds, str(out_path))
    finally:
        # Cubre el final normal y la interrupción con Ctrl-C: el último estado
        # completo queda persistido sin tocar el checkpoint de entrada.
        if save_path is not None and steps_done > 0:
            save_checkpoint(save_path, model, optimizer, cfg, meta={'steps': steps_done, 'loss': mean_loss})
