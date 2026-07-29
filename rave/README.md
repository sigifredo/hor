# hor.rave — fase 1 + fase 2 adversarial

Refactor sobre la versión anterior. Los cambios corrigen el colapso posterior
observado en la corrida de 200k pasos y añaden discriminador multi-escala.

## Qué cambió

**Anti-colapso en fase 1:**

- `beta_max` default bajado de `0.1` a `1e-4` (factor 1000).
- `warmup_steps` default subido de `10_000` a `100_000` (factor 10).
- Nueva opción `free_bits` (default `0.1` nats/dim), floor por dimensión de la KL.
- `RAVELoss` ahora reporta `kl` (con clamp) y `kl_raw` (sin clamp) para diagnóstico.

**Nuevo módulo `discriminator.py`:**

- `MultiScaleDiscriminator`: 3 escalas por AvgPool1d, MelGAN-style.
- `ScaleDiscriminator`: 5 capas conv1d con `weight_norm` y grupos progresivos.
- Sin BatchNorm (weight_norm es la única normalización).
- ~4.35 M params totales, dimensionado para 6 GB VRAM con batch=4.

**Nuevas funciones en `losses.py`:**

- `hinge_loss_d`: hinge estándar para el D.
- `hinge_loss_g`: hinge non-saturating para el G.
- `feature_matching_loss`: L1 sobre features intermedios del D.
- `kl_free_bits`: KL con clamp por dimensión, devuelve `(kl_optimized, kl_raw)`.

**`engine.py` refactorizado en dos fases:**

- `_train_phase1`: VAE puro con las correcciones anti-colapso.
- `_train_phase2`: alterna D→G en cada paso, con encoder frozen por default.
- Checkpoints de fase 2 incluyen `discriminator` y `optimizer_d`.
- Nombres de checkpoint distintos: `rave_p1_stepXXXX.pt` vs `rave_p2_stepXXXX.pt`.

**CLI:**

- `--phase {1,2}` en `train`.
- Argumentos nuevos: `--free-bits`, `--freeze-encoder`, `--d-lr`, `--adversarial-weight`, `--fm-weight`, `--d-grad-clip`.
- Defaults ajustados en `--beta-max` y `--warmup-steps` como se describe arriba.

## Cómo usar para tu corpus

### Fase 1 (~100k pasos, ~3 h en tu GPU)

```bash
python -m rave train --phase 1 \
    --audio /ruta/a/tu_mezcla_16min.wav \
    --out-dir runs/rave_v2_p1 \
    --n-steps 100000 \
    --batch-size 4 \
    --beta-max 1e-4 \
    --warmup-steps 100000 \
    --free-bits 0.1
```

Diagnóstico durante la corrida: revisa `train_log.csv`. La señal buena es
`kl_raw` estabilizándose entre `0.5` y `5` por dimensión (o sea, KL total
entre `32` y `320` con `latent_dim=64`). Si `kl_raw` cae por debajo de
`0.1`/dim, sigue habiendo colapso: baja más `beta_max` o sube `free_bits`.

### Fase 2 (~50-100k pasos)

```bash
python -m rave train --phase 2 \
    --audio /ruta/a/tu_mezcla_16min.wav \
    --out-dir runs/rave_v2_p2 \
    --resume-from runs/rave_v2_p1/checkpoints/rave_p1_final.pt \
    --resume-mode finetune \
    --n-steps 100000 \
    --batch-size 4 \
    --adversarial-weight 1.0 \
    --fm-weight 10.0
```

Con `--resume-mode finetune` los optimizadores arrancan desde cero y solo se
cargan los pesos del modelo. Es lo correcto al cambiar de fase.

### Diagnóstico en fase 2

Señales de salud:

- `d_loss` oscila alrededor de `~1.0-2.0` sin colapsar a `0` ni divergir.
- `g_adv` cerca de `0` cuando D no distingue; si `g_adv >> 0` sostenido, D está ganando (baja `d_lr`); si `g_adv << 0` sostenido, G está ganando (sube `d_lr` o `adversarial_weight`).
- `fm` desciende gradualmente.
- `stft_sc` sigue bajando por debajo de lo que lograste en fase 1.

Modos de fallo a vigilar:

- **Adversarial artifacts:** tonos hi-fi puntuales sobre reconstrucciones borrosas. Sube `--fm-weight` (prueba `20` o `50`).
- **Mode collapse en G:** todas las salidas suenan igual. Reduce `--adversarial-weight` (`0.5`) o revisa que el encoder no esté frozen si tu latente aún no está bien.
- **D imbatible:** `d_loss → 0` en pocos miles de pasos. Reduce `d_lr` a `5e-5` o `2e-5`.

## Ejecutar los tests

```bash
cd /donde/vive/hor
python -m pytest rave/ -v
```

Los tests cubren: PQMF, modelo, losses de fase 1 y 2, discriminador,
smoke tests end-to-end (fase 1, fase 2, y transición finetune).

## Archivos del paquete

```
rave/
    __init__.py           # exports
    __main__.py           # CLI con --phase
    data.py               # dataset (sin cambios)
    discriminator.py      # NUEVO
    engine.py             # REFACTORIZADO en dos fases
    generate.py           # info muestra fase 2 si aplica
    losses.py             # EXTENDIDO: free bits, hinge, feature matching
    model.py              # sin cambios
    pqmf.py               # sin cambios
    test_discriminator.py # NUEVO
    test_e2e.py           # NUEVO (smoke)
    test_losses.py        # EXTENDIDO
    test_model.py         # sin cambios funcionales
    test_pqmf.py          # sin cambios funcionales
```
