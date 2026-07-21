'''Persistencia del estado de entrenamiento.

Un checkpoint captura lo necesario para continuar el entrenamiento online sin
partir de cero: pesos del modelo, momentos de Adam y metadatos. Los momentos
del optimizador se guardan porque un warm start solo con pesos obliga a Adam a
reestimar su escala adaptativa durante los primeros cientos de pasos.

La arquitectura se guarda junto al estado para detectar temprano un desajuste
con el Config vigente (mensaje claro en vez de un size mismatch críptico si el
Config cambió después de generar el checkpoint).'''

import torch

from .config import Config

_ARCH_FIELDS = (
    'quant_levels',
    'n_layers',
    'residual_channels',
    'dilation_channels',
    'skip_channels',
    'kernel_size',
    'sample_rate',
)


def _arch(cfg: Config) -> dict:
    return {f: getattr(cfg, f) for f in _ARCH_FIELDS}


def save_checkpoint(path, model, optimizer, cfg: Config, meta: dict | None = None) -> None:
    torch.save(
        {
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'arch': _arch(cfg),
            'meta': dict(meta or {}),
        },
        path,
    )


def load_checkpoint(path, model, optimizer=None, cfg: Config | None = None) -> dict:
    '''Carga pesos en `model` (y estado de Adam en `optimizer`, si se pasa).
    Con `cfg` valida la arquitectura. Devuelve los metadatos del checkpoint.'''
    ckpt = torch.load(path, map_location='cpu', weights_only=True)

    if cfg is not None and ckpt.get('arch') != _arch(cfg):
        raise ValueError(f'arquitectura del checkpoint no coincide con el Config vigente: {ckpt.get("arch")} != {_arch(cfg)}')

    model.load_state_dict(ckpt['model'])

    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt.get('meta', {})
