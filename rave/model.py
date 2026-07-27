'''VAE tipo RAVE: encoder y decoder convolucionales sobre subbandas PQMF.

Estructura del forward:

    audio (B, 1, T)
        → PQMF.analyze          → (B, n_bands, T / n_bands)
        → Encoder               → mu, log_sigma at (B, latent_dim, T / S_total)
        → reparametrización     → z at (B, latent_dim, T / S_total)
        → Decoder               → subbandas (B, n_bands, T / n_bands)
        → PQMF.synthesize       → audio (B, 1, T)

    donde S_total = n_bands * prod(strides).

Convenciones:
    * Cada EncoderBlock consiste en una pila residual seguida de convolución
      diezmadora con kernel = 2 * stride y padding = stride // 2. Esto garantiza
      output_len = input_len / stride cuando input_len es múltiplo del stride.

    * Cada DecoderBlock aplica primero ConvTranspose1d con kernel = 2 * upsample,
      stride = upsample, padding = upsample // 2, produciendo output_len =
      input_len * upsample. Después inyecta ruido con escala aprendida por
      canal (inicializada a cero) y aplica una pila residual.

    * La ResidualStack apila n_res_per_block unidades con dilataciones 3^i,
      dando un campo receptivo efectivo de sum(2 * 3^i) por bloque.

    * Los canales se duplican en cada EncoderBlock y se dividen a la mitad en
      cada DecoderBlock, siguiendo la convención de RAVE original.
'''

from __future__ import annotations


from .pqmf import PQMF
from torch.nn.utils.parametrizations import weight_norm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _wn(module: nn.Module) -> nn.Module:
    '''weight_norm sobre convoluciones, envoltorio local.'''
    return weight_norm(module)


class ResidualUnit(nn.Module):
    '''Unidad residual: LeakyReLU → Conv3(d) → LeakyReLU → Conv1 → +x.'''

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv1 = _wn(nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation, padding=dilation))
        self.conv2 = _wn(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(F.leaky_relu(x, 0.2))
        y = self.conv2(F.leaky_relu(y, 0.2))
        return x + y


class ResidualStack(nn.Module):
    '''Pila de unidades residuales con dilataciones 1, 3, 9, ...'''

    def __init__(self, channels: int, n_units: int = 3):
        super().__init__()
        self.units = nn.ModuleList([ResidualUnit(channels, dilation=3**i) for i in range(n_units)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for u in self.units:
            x = u(x)
        return x


class NoiseInjection(nn.Module):
    '''Añade ruido gaussiano con escala aprendida por canal.

    La escala se inicializa a cero: al comienzo del entrenamiento el ruido no
    contribuye, y el modelo lo activa solo cuando le resulta útil (convención
    tomada de StyleGAN).
    '''

    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x)
        return x + self.scale * noise


class EncoderBlock(nn.Module):
    '''Bloque de encoder: ResidualStack + conv diezmadora.

    Diseño de padding para que output_len = input_len // stride exactamente:
        kernel_size = 2 * stride
        padding     = stride // 2

    Verificación algebraica:
        L_out = (L_in + 2 * (stride//2) - 2*stride) // stride + 1
              = (L_in + stride - 2*stride) // stride + 1  (para stride par)
              = (L_in - stride) // stride + 1
              = L_in // stride  (si L_in múltiplo de stride)
    '''

    def __init__(self, c_in: int, c_out: int, stride: int, n_res: int = 3):
        super().__init__()
        self.res = ResidualStack(c_in, n_res)
        self.down = _wn(nn.Conv1d(c_in, c_out, kernel_size=2 * stride, stride=stride, padding=stride // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.res(x)
        x = F.leaky_relu(x, 0.2)
        return self.down(x)


class DecoderBlock(nn.Module):
    '''Bloque de decoder: ConvTranspose1d + inyección de ruido + ResidualStack.

    Con kernel = 2 * upsample, stride = upsample, padding = upsample // 2:
        L_out = (L_in - 1) * upsample - 2 * (upsample//2) + 2 * upsample
              = L_in * upsample  (para upsample par)
    '''

    def __init__(self, c_in: int, c_out: int, upsample: int, n_res: int = 3):
        super().__init__()
        self.up = _wn(nn.ConvTranspose1d(c_in, c_out, kernel_size=2 * upsample, stride=upsample, padding=upsample // 2))
        self.noise = NoiseInjection(c_out)
        self.res = ResidualStack(c_out, n_res)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(x, 0.2)
        x = self.up(x)
        x = self.noise(x)
        return self.res(x)


class Encoder(nn.Module):
    '''Encoder: n_bands → 2 * latent_dim, con strides definidos.

    Devuelve (mu, log_sigma), cada uno de forma (B, latent_dim, L_z) donde
    L_z = T_in / prod(strides).
    '''

    def __init__(self, n_bands: int, hidden_channels: int, strides: tuple[int, ...], latent_dim: int, n_res_per_block: int = 3):
        super().__init__()
        self.input_conv = _wn(nn.Conv1d(n_bands, hidden_channels, kernel_size=7, padding=3))
        blocks = []
        c = hidden_channels
        for stride in strides:
            c_next = c * 2
            blocks.append(EncoderBlock(c, c_next, stride, n_res_per_block))
            c = c_next
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = _wn(nn.Conv1d(c, 2 * latent_dim, kernel_size=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_conv(x)
        for block in self.blocks:
            x = block(x)
        params = self.out_conv(x)
        mu, log_sigma = params.chunk(2, dim=1)
        return mu, log_sigma


class Decoder(nn.Module):
    '''Decoder: latent_dim → n_bands, upsampling en cada bloque.

    Los strides se recorren invertidos respecto al encoder para reconstruir
    exactamente la resolución temporal de las subbandas.
    '''

    def __init__(self, latent_dim: int, hidden_channels: int, strides: tuple[int, ...], n_bands: int, n_res_per_block: int = 3):
        super().__init__()
        c = hidden_channels * (2 ** len(strides))
        self.input_conv = _wn(nn.Conv1d(latent_dim, c, kernel_size=7, padding=3))
        blocks = []
        for upsample in reversed(strides):
            c_next = c // 2
            blocks.append(DecoderBlock(c, c_next, upsample, n_res_per_block))
            c = c_next
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = _wn(nn.Conv1d(c, n_bands, kernel_size=7, padding=3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.input_conv(z)
        for block in self.blocks:
            z = block(z)
        return self.out_conv(z)


class RAVE(nn.Module):
    '''VAE tipo RAVE con PQMF integrado.

    Args:
        n_bands: número de subbandas del PQMF.
        pqmf_taps: taps del prototipo PQMF (par).
        hidden_channels: canales del primer bloque del encoder.
        strides: factores de diezmado por bloque del encoder. El decoder los
            aplica en orden inverso.
        latent_dim: dimensión del espacio latente.
        n_res_per_block: unidades residuales por ResidualStack.
        log_sigma_range: recorte del logaritmo de sigma, evita explosiones
            numéricas en la reparametrización y en la KL.
    '''

    def __init__(self, n_bands: int = 16, pqmf_taps: int = 126, hidden_channels: int = 64, strides: tuple[int, ...] = (2, 4, 2), latent_dim: int = 64, n_res_per_block: int = 3, log_sigma_range: tuple[float, float] = (-10.0, 2.0)):
        super().__init__()
        self.pqmf = PQMF(n_bands=n_bands, taps=pqmf_taps)
        self.encoder = Encoder(n_bands, hidden_channels, strides, latent_dim, n_res_per_block)
        self.decoder = Decoder(latent_dim, hidden_channels, strides, n_bands, n_res_per_block)
        self.latent_dim = latent_dim
        self.n_bands = n_bands
        self.strides = tuple(strides)
        self.total_stride = n_bands * math.prod(strides)
        self.log_sigma_min = log_sigma_range[0]
        self.log_sigma_max = log_sigma_range[1]

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''(B, 1, T) → (mu, log_sigma) at (B, latent_dim, T / total_stride).'''
        if x.size(-1) % self.total_stride != 0:
            raise ValueError(f'longitud {x.size(-1)} no es múltiplo de total_stride=' f'{self.total_stride}')
        bands = self.pqmf.analyze(x)
        mu, log_sigma = self.encoder(bands)
        log_sigma = log_sigma.clamp(min=self.log_sigma_min, max=self.log_sigma_max)
        return mu, log_sigma

    def reparameterize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        '''(B, latent_dim, L_z) → audio (B, 1, L_z * total_stride).'''
        bands = self.decoder(z)
        return self.pqmf.synthesize(bands)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''Pasada completa. Devuelve (x_hat, mu, log_sigma).'''
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        x_hat = self.decode(z)
        return x_hat, mu, log_sigma

    @torch.no_grad()
    def sample_prior(self, batch_size: int, length: int, device: torch.device | None = None) -> torch.Tensor:
        '''Muestrea z ~ N(0, I) y devuelve audio (B, 1, length * total_stride).

        Con dataset pequeño el prior N(0, I) no cubre el manifold entrenado y
        las muestras suelen ser pobres. Este método existe principalmente para
        diagnóstico.
        '''
        device = device or next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim, length, device=device)
        return self.decode(z)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        '''Reconstrucción determinista (usa mu en vez de sample).'''
        mu, _ = self.encode(x)
        return self.decode(mu)

    def parameter_count(self) -> dict:
        counts = {'encoder': 0, 'decoder': 0, 'pqmf': 0}
        for name, p in self.named_parameters():
            top = name.split('.')[0]
            counts[top] = counts.get(top, 0) + p.numel()
        counts['total'] = sum(counts.values())
        return counts
