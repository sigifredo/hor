'''Tests del discriminador multi-escala.'''

from __future__ import annotations
from .discriminator import MultiScaleDiscriminator, ScaleDiscriminator

import pytest
import torch


def test_scale_discriminator_shapes():
    d = ScaleDiscriminator()
    x = torch.randn(2, 1, 32_768)
    features, logit = d(x)

    assert isinstance(features, list)
    assert len(features) == 5, f'esperados 5 feature maps, hay {len(features)}'
    for f in features:
        assert f.dim() == 3, f'feature debe ser (B, C, T), es {f.shape}'
        assert f.size(0) == 2

    assert logit.dim() == 3
    assert logit.size(0) == 2
    assert logit.size(1) == 1


def test_multi_scale_shapes():
    msd = MultiScaleDiscriminator(n_scales=3)
    x = torch.randn(2, 1, 32_768)
    outputs = msd(x)

    assert len(outputs) == 3
    for feats, logit in outputs:
        assert isinstance(feats, list)
        assert logit.size(0) == 2


def test_multi_scale_gradients_flow():
    msd = MultiScaleDiscriminator(n_scales=3)
    x = torch.randn(2, 1, 8192, requires_grad=True)
    outputs = msd(x)
    logits = [l for _, l in outputs]
    loss = sum(l.mean() for l in logits)
    loss.backward()

    grads_ok = 0
    for p in msd.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads_ok += 1
    assert grads_ok > 0, 'ningún parámetro del D recibió gradiente'


def test_parameter_count_under_budget():
    '''Con la configuración default, D debe estar por debajo de 5 M params.'''
    msd = MultiScaleDiscriminator(n_scales=3)
    total = sum(p.numel() for p in msd.parameters())
    assert total < 5_000_000, f'D tiene {total:,} params, excede el presupuesto'
    assert total > 3_000_000, f'D tiene solo {total:,} params, sospechosamente bajo'


def test_scale_discriminator_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match='mismo largo'):
        ScaleDiscriminator(channels=(16, 64), kernel_sizes=(15,), strides=(1, 4), groups=(1, 4))
