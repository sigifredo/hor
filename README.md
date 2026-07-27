# hor

```bash
python pretrain.py --input assets/audio.wav --steps 5000 --device cuda --out base.pt
python main.py --input assets/audio.wav --mode batched --checkpoint base.pt --save-checkpoint sesion1.pt --save-every 10 --device cuda
python generate.py --checkpoint sesion1.pt --duration 60 --temperature 0.85 --out pieza.wav
```

## Rave

### Run

```bash
python -m rave train --audio <in audio> --out-dir resources/runs/rave_v1 --n-steps 100000
```

## Install torch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

# Comandos del subpaquete `hor.rave`

Referencia rápida de los subcomandos. Todos se invocan como módulo. La ruta
del paquete asume que trabajas desde el directorio padre de `rave/`.

## Entrenamiento

### Entrenamiento básico

Con los defaults (100.000 pasos, batch 8, latent_dim 64, β_max 0.1, warmup
10.000 pasos, checkpoints cada 5.000 pasos).

```bash
python -m rave train --audio corpus.wav --out-dir runs/rave_v1
```

### Entrenamiento con parámetros custom

Ejemplo típico ajustando duración y frecuencia de checkpoints:

```bash
python -m rave train \
    --audio corpus.wav \
    --out-dir runs/rave_v1 \
    --n-steps 50000 \
    --batch-size 16 \
    --lr 1e-4 \
    --ckpt-every 2000
```

### Continuar el mismo entrenamiento

Retomar donde quedó, con el mismo audio y el mismo estado del optimizador.
Útil si el proceso se cortó o quieres extender más pasos sobre el mismo
material.

```bash
python -m rave train \
    --audio corpus.wav \
    --out-dir runs/rave_v1_ext \
    --resume-from runs/rave_v1/checkpoints/rave_final.pt \
    --resume-mode continue \
    --n-steps 100000
```

En modo `continue`:

- Los pesos, el estado de Adam y el step global se restauran.
- El β del KL ya está en su máximo (warmup consumido).
- Los nuevos checkpoints se numeran a partir del step global.

### Fine-tuning en audio nuevo

Cargar los pesos del checkpoint pero entrenar sobre otro corpus. El
optimizador arranca desde cero para no arrastrar estadísticas de gradientes
del audio anterior. Conviene reducir el learning rate y el warmup del KL.

```bash
python -m rave train \
    --audio audio_nuevo.wav \
    --out-dir runs/rave_v1_ft \
    --resume-from runs/rave_v1/checkpoints/rave_final.pt \
    --resume-mode finetune \
    --lr 1e-5 \
    --warmup-steps 2000 \
    --n-steps 50000
```

En modo `finetune`:

- Solo los pesos del modelo se restauran.
- El optimizador y el step global arrancan en cero.
- El warmup del KL se aplica de nuevo desde el step 1.

## Inspección de un checkpoint

Lee los metadatos sin instanciar el modelo. Devuelve tamaño, step,
arquitectura, hiperparámetros de entrenamiento y rutas del corpus original.

```bash
python -m rave info \
    --checkpoint runs/rave_v1/checkpoints/rave_final.pt
```

Antes de reanudar un entrenamiento con `--resume-from`, este comando permite
verificar los valores de la arquitectura original (n_bands, strides,
latent_dim, etc.) para pasarlos idénticos y evitar mismatch al cargar el
state_dict.

## Reconstrucción

Toma un audio, lo codifica al latente y lo decodifica. La salida es una
versión "vista a través" del modelo.

```bash
python -m rave reconstruct \
    --checkpoint runs/rave_v1/checkpoints/rave_final.pt \
    --audio entrada.wav \
    --output reconstruido.wav
```

## Muestreo desde el prior

Genera audio desde ruido gaussiano decodificado. No hay entrada de audio.

```bash
python -m rave sample \
    --checkpoint runs/rave_v1/checkpoints/rave_final.pt \
    --duration 8.0 \
    --output prior.wav
```

Con `--seed` la salida es reproducible entre corridas:

```bash
python -m rave sample \
    --checkpoint runs/rave_v1/checkpoints/rave_final.pt \
    --duration 8.0 \
    --seed 42 \
    --output prior.wav
```

## Notas prácticas

### Arquitectura consistente al reanudar

Al usar `--resume-from`, los argumentos de arquitectura pasados deben
coincidir con los del checkpoint. Si el checkpoint fue entrenado con
`latent_dim=64` y `strides=[2, 4, 2]`, hay que pasar los mismos valores o
`load_state_dict` fallará con mismatch de shapes. Usa `info` para
consultarlos.

### Carpeta de salida por corrida

Recomendable usar `--out-dir` distinto en cada corrida. Los CSV de logging
hacen append si el archivo existe, y los checkpoints con `keep_last_ckpts=3`
pueden borrar los anteriores en la misma carpeta al hacer pruning.

### Fine-tuning como material expresivo

En fine-tuning con audio contrastante, los primeros 100-500 pasos suelen
producir los artefactos más ricos porque los pesos aún conservan residuos
del corpus original. Guardar checkpoints frecuentes al inicio permite
explorar esa transición:

```bash
python -m rave train \
    --audio audio_nuevo.wav \
    --out-dir runs/rave_v1_ft \
    --resume-from runs/rave_v1/checkpoints/rave_final.pt \
    --resume-mode finetune \
    --lr 1e-5 \
    --warmup-steps 2000 \
    --n-steps 5000 \
    --ckpt-every 250
```

### Prior sampling con corpus pequeño

Con archivo único, el latente entrenado ocupa una variedad muy estrecha
dentro de N(0, I) y muestrear la gaussiana completa cae mayoritariamente
fuera de esa variedad. El decoder produce algo que promedia el corpus. Es
comportamiento esperado del método, no un fallo de implementación.

### Consumo de tiempo

Los defaults (`--n-steps 100000`) requieren varias horas en GPU consumer.
Para iteración rápida durante desarrollo:

```bash
python -m rave train \
    --audio corpus.wav \
    --out-dir runs/rave_test \
    --n-steps 5000 \
    --warmup-steps 500 \
    --ckpt-every 500 \
    --log-every 50
```
