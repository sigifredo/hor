# realtime_wavenet

WaveNet causal dilatado con **entrenamiento online real sobre audio crudo** y
**generación autoregresiva** ejecutándose de forma concurrente. El sistema
escucha una fuente, entrena por rondas mientras suena, y publica los pesos
nuevos mediante un swap atómico que la generación recoge en caliente.

Proyecto de corte artístico y experimental. El objetivo no es convergencia
estable sino observar el comportamiento del acoplamiento entrada → entrenamiento
→ generación. Fenómenos como el olvido catastrófico o los transitorios en cada
commit se tratan como material expresivo, no como fallas a corregir.

## Arquitectura

```
fuente ──read(n)──> acumulador ──chunk μ-law──> [cola acotada] ──> entrenamiento
                                                                        │
                                                          swap atómico de pesos
                                                                        ▼
                        sink <──audio── generación <──puntero de pesos── (shared)
```

Tres hilos sobre estructuras compartidas:

- **Acumulador**: `source.read(n)` → codifica μ-law → arma chunks de 16 000
  muestras → los encola. La cola acotada impone contrapresión: con archivo, el
  ritmo lo marca el _throughput_ de entrenamiento, no el reloj de pared.
- **Entrenamiento**: consume un chunk, `forward`+`backward`, paso de Adam sobre
  un modelo **persistente** (warm start + estado de Adam conservado entre
  rondas), clona una copia congelada y hace el swap del puntero.
- **Generación**: relee el puntero por bloque, produce muestras con caché
  incremental por capa (_fast-wavenet_), las escribe al sink.

El puntero de pesos usa **dos instancias**: el modelo de entrenamiento nunca lo
lee la generación; tras cada ronda se publica una copia fresca. La generación ve
pesos viejos o nuevos, jamás un estado a medias. Las cachés de generación
**persisten** a través del swap: cada commit introduce un micro-transitorio
audible, que es el instante perceptible de "aprendizaje".

## Modelo (dimensiones definitivas)

| parámetro                              | valor                                     |
| -------------------------------------- | ----------------------------------------- |
| tasa de muestreo                       | 16 kHz, mono                              |
| cuantización                           | μ-law, 256 niveles (categórica + softmax) |
| pilas                                  | 1                                         |
| capas dilatadas                        | 10 (dilataciones 1,2,…,512)               |
| kernel                                 | 2                                         |
| canales residuales / dilatación / skip | 64 / 64 / 128                             |
| campo receptivo                        | 1024 muestras (64 ms)                     |
| parámetros                             | ≈ 0.35 M                                  |

Dos horizontes de memoria distintos coexisten: el **campo receptivo** (fijo,
arquitectónico, 64 ms) y la **deriva de pesos** vía warm start (aprendida, a
escala de minutos/sesión).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # sounddevice solo hace falta para --mode live
```

## Uso

```bash
# Material de prueba (opcional)
python examples/make_test_audio.py test.wav

# Modo archivo: ejecuta 20 s y escribe la salida (funciona headless)
python -m realtime_wavenet.main --input test.wav --mode file \
    --duration 20 --output out.wav

# Modo vivo: reproduce en tiempo real (requiere sounddevice + dispositivo)
python -m realtime_wavenet.main --input test.wav --mode live

# Control expresivo de temperatura
python -m realtime_wavenet.main --input test.wav --temperature 1.3
```

## Extensión: nuevas fuentes de entrada

El core depende solo del puerto `AudioSource` (`sources.py`). Agregar una fuente
(micrófono en vivo, sensores, imágenes transducidas) es:

1. Subclasear `AudioSource` e implementar `read(n)` devolviendo float32 mono en
   [-1, 1] a 16 kHz (el adaptador resamplea/downmixea internamente).
2. Registrarla en `_SOURCES`.

Ni el acumulador, ni el entrenamiento, ni la generación cambian. Para una fuente
en vivo (push, clockeada por hardware), el ring buffer de entrada y el callback
de audio viven **dentro** del adaptador; `read(n)` bloquea drenándolo.

## Limitaciones conocidas (por diseño, no bugs)

- **Velocidad de generación**: a 16 kHz hay que emitir una muestra cada 62.5 µs
  en promedio sostenido. La generación autoregresiva muestra a muestra en
  PyTorch eager tiene un overhead por paso que **muy plausiblemente no alcanza
  ese ritmo** en CPU. Es el punto que hay que medir primero. Si no cierra:
  kernels de inferencia dedicados, generar con más latencia de salida, o bajar
  la tasa efectiva. En `--mode file` la generación corre tan rápido como pueda y
  el resultado es correcto aunque no sea tiempo real; el modo `live` revela el
  déficit como underruns.
- **GIL**: los tres hilos comparten intérprete. Las ops de PyTorch liberan el
  GIL, pero el callback de audio en Python lo retoma cada periodo. Escape
  estructural si la contención es audible: `multiprocessing` con memoria
  compartida. Esta es la v1 con `threading` + buffers generosos.
- **Olvido**: cada ronda entrena solo con la ventana más reciente; el warm start
  conserva algo del pasado pero no hay mecanismo explícito (EWC, replay) contra
  el olvido catastrófico. Elección deliberada.
- **Sink en vivo**: el callback adquiere un lock breve. Bajo el GIL una ruta de
  audio verdaderamente lock-free en Python no es alcanzable.

## Estructura

```
realtime_wavenet/
  config.py     # hiperparámetros definitivos
  mu_law.py     # compansión μ-law
  model.py      # WaveNet (forward paralelo) + WaveNetGenerator (incremental)
  sources.py    # puerto AudioSource + FileLoopSource + fábrica
  sinks.py      # puerto AudioSink + FileSink + LiveSink
  engine.py     # SharedWeights + loops + Engine
  main.py       # CLI
examples/
  make_test_audio.py
```
