'''Subpaquete hor.rave: VAE tipo RAVE para síntesis de audio.

Fases:
    * Fase 1: VAE puro con anti-colapso (STFT + L1 + KL con free bits).
    * Fase 2: VAE + discriminador multi-escala (hinge + feature matching).
'''

from .pqmf import PQMF
from .model import RAVE
from .discriminator import MultiScaleDiscriminator, ScaleDiscriminator
from .losses import (
    RAVELoss,
    MultiScaleSTFTLoss,
    kl_diagonal_gaussian,
    kl_free_bits,
    hinge_loss_d,
    hinge_loss_g,
    feature_matching_loss,
)
from .engine import TrainConfig, train
from .generate import load_model, reconstruct, sample_prior

__all__ = [
    'PQMF',
    'RAVE',
    'MultiScaleDiscriminator',
    'ScaleDiscriminator',
    'RAVELoss',
    'MultiScaleSTFTLoss',
    'kl_diagonal_gaussian',
    'kl_free_bits',
    'hinge_loss_d',
    'hinge_loss_g',
    'feature_matching_loss',
    'TrainConfig',
    'train',
    'load_model',
    'reconstruct',
    'sample_prior',
]
