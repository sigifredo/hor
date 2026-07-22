'''Hiperparámetros definitivos del sistema.

Todas las dimensiones del modelo son fijas y derivan de decisiones de diseño ya
cerradas. Los parámetros de sistema (chunk_len, queue_maxsize, gen_block,
blocksize) son perillas de ejecución: su valor óptimo depende del wall-clock
medido en la máquina concreta, no de la arquitectura.'''

import dataclasses


@dataclasses.dataclass(frozen=True)
class Config:
    # --- Representación de la señal ---
    sample_rate: int = 16000  # mono
    quant_levels: int = 256  # niveles mu-law (8 bits)
    mu: int = 255  # parámetro de compansión mu-law

    # --- Núcleo del modelo (definitivo) ---
    n_layers: int = 16  # dilataciones 1,2,...,512
    residual_channels: int = 64
    dilation_channels: int = 64  # canales de filtro y compuerta
    skip_channels: int = 128
    kernel_size: int = 2

    # --- Entrenamiento por ronda ---
    chunk_len: int = 16000  # 1 s a 16 kHz
    lr: float = 1e-3
    queue_maxsize: int = 2  # contrapresión: pacing por throughput

    # --- Generación ---
    gen_block: int = 256  # muestras por relectura del puntero de pesos
    temperature: float = 1.0  # >1 dispersa, <1 concentra
    out_capacity: int = 16000  # capacidad del ring de salida (LiveSink)
    blocksize: int = 1024  # frames por callback de salida (LiveSink)

    # --- Cómputo ---
    device: str = 'cpu'  # 'cpu' o 'cuda' (o 'cuda:0', etc.)

    @property
    def dilations(self) -> list:
        return [2**i for i in range(self.n_layers)]

    @property
    def receptive_field(self) -> int:
        # kernel 2: cada capa aporta su dilatación; +1 por la muestra actual.
        return sum(self.dilations) + 1  # 1024 muestras = 64 ms a 16 kHz

    @property
    def loss_warmup(self) -> int:
        # Posiciones iniciales descartadas de la pérdida (contexto incompleto).
        return self.receptive_field
