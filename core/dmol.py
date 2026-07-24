'''Discretized Mixture of Logistics (DMoL) para audio a 16 bits.

Estándar de PixelCNN++/WaveNet++. En vez de predecir una categórica sobre 256
clases (mu-law), el modelo predice K componentes de una mezcla de logísticas
sobre la señal continua en [-1, 1]. Al muestrear, se discretiza a 16 bits
para conservar la naturaleza cuantizada del formato de salida, pero la
distribución sigue siendo continua durante el entrenamiento.

Ventajas frente a categórica mu-law:
  - Sin ruido de cuantización de 8 bits en la salida.
  - Loss suave sobre el continuo, no cross-entropy sobre 256 clases.
  - K componentes permiten modelar multimodalidad (útil en transitorios).

Parámetros de salida por muestra: 3*K
  - K logits de mezcla (pi, sin softmax; se aplica en la loss)
  - K medias mu en [-1, 1]
  - K log-escalas log_s (clamped inferior para estabilidad)

Referencia: Salimans et al. 2017, PixelCNN++.'''

import torch
import torch.nn.functional as F

LOG_SCALE_MIN = -7.0  # log_s < -7 satura y produce gradientes cero: no vale la pena
BITS = 16  # cuantización de salida
LEVELS = 2**BITS  # 65536
HALF_BIN = 1.0 / (LEVELS - 1)  # media anchura de bin en [-1, 1]


def n_params(n_mix: int) -> int:
    '''Número de parámetros de salida por muestra.'''
    return 3 * n_mix


def _split_params(params: torch.Tensor, n_mix: int):
    '''(B, 3K, T) -> (pi_logits, mu, log_s), cada uno (B, K, T).'''
    pi = params[:, :n_mix, :]
    mu = params[:, n_mix : 2 * n_mix, :]
    log_s = params[:, 2 * n_mix : 3 * n_mix, :]
    log_s = torch.clamp(log_s, min=LOG_SCALE_MIN)
    # Acotamos mu a [-1, 1] con tanh: mantiene los centros dentro del rango
    # de la señal y estabiliza el entrenamiento (Salimans et al. 2017).
    mu = torch.tanh(mu)
    return pi, mu, log_s


def dmol_loss(params: torch.Tensor, target: torch.Tensor, n_mix: int) -> torch.Tensor:
    '''Negative log-likelihood de la señal continua bajo la mezcla, discretizada.

    params: (B, 3K, T)  parámetros crudos de salida
    target: (B, T)      señal en [-1, 1] float
    return: escalar     NLL promedio por muestra, en nats

    Para cada muestra, la probabilidad de x bajo la mezcla es
        sum_k pi_k * (CDF_logistic((x + hb - mu_k)/s_k) - CDF_logistic((x - hb - mu_k)/s_k))
    donde CDF_logistic(z) = sigmoid(z). En los bordes (x = ±1) usamos la cola
    correspondiente para no perder masa.'''
    pi_logits, mu, log_s = _split_params(params, n_mix)
    x = target.unsqueeze(1)  # (B, 1, T) - broadcast contra K

    inv_s = torch.exp(-log_s)
    centered = x - mu  # (B, K, T)

    plus = inv_s * (centered + HALF_BIN)
    minus = inv_s * (centered - HALF_BIN)

    cdf_plus = torch.sigmoid(plus)
    cdf_minus = torch.sigmoid(minus)

    # Probabilidad del bin discreto = CDF(centro + hb) - CDF(centro - hb).
    # Log-numérico estable: en la región central usamos log(cdf_plus - cdf_minus)
    # con clamp a 1e-12; en los bordes usamos las colas.
    #
    # Aproximación mid_in_log evita underflow cuando ambos CDFs son casi iguales
    # (varianza mucho menor que HALF_BIN); ahí usamos log(pdf * 2*hb).
    log_cdf_plus = plus - F.softplus(plus)  # log P(X < x + hb)
    log_one_minus_cdf_minus = -F.softplus(minus)  # log P(X > x - hb)
    cdf_delta = cdf_plus - cdf_minus

    # log(cdf_delta) con clamp para estabilidad. Cuando cdf_delta subdesborda a 0,
    # significa que |x - mu| >> s (target lejos del centro), donde la masa del bin
    # es genuinamente ~0. El clamp a 1e-12 => log = -27.6, que es el "castigo" que
    # queremos para gradientes que empujen mu hacia x o hagan s más grande.
    log_probs_central = torch.log(torch.clamp(cdf_delta, min=1e-12))

    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,  # cola izquierda
        torch.where(
            x > 0.999,
            log_one_minus_cdf_minus,  # cola derecha
            log_probs_central,
        ),
    )

    # Loss = -log sum_k pi_k * p_k, con log-sum-exp estable
    log_pi = F.log_softmax(pi_logits, dim=1)  # (B, K, T)
    log_mix = torch.logsumexp(log_pi + log_probs, dim=1)  # (B, T)
    return -log_mix.mean()


def dmol_sample(params: torch.Tensor, n_mix: int, temperature: float = 1.0) -> torch.Tensor:
    '''Muestrea de la mezcla; devuelve señal en [-1, 1] cuantizada a 16 bits.

    params: (B, 3K, T)
    return: (B, T)  float32 en {-1, -1 + 1/(L-1), ..., 1}

    Proceso:
      1. Escoger componente k ~ softmax(pi_logits / T)
      2. Muestrear de logística(mu_k, s_k) via inversión de CDF
      3. Cuantizar a 16 bits, clip a [-1, 1]'''
    pi_logits, mu, log_s = _split_params(params, n_mix)
    B, K, T = pi_logits.shape

    # Selección de componente con Gumbel-max (equivalente a multinomial de softmax)
    if temperature <= 0:
        k_idx = pi_logits.argmax(dim=1, keepdim=True)  # (B, 1, T)
    else:
        gumbel = -torch.log(-torch.log(torch.rand_like(pi_logits).clamp_(1e-9, 1 - 1e-9)))
        k_idx = (pi_logits / temperature + gumbel).argmax(dim=1, keepdim=True)

    mu_k = mu.gather(1, k_idx).squeeze(1)  # (B, T)
    log_s_k = log_s.gather(1, k_idx).squeeze(1)

    # Inversión de CDF logística: F(x) = sigmoid((x - mu)/s), F^-1(u) = mu + s * (log u - log(1-u))
    u = torch.rand_like(mu_k).clamp_(1e-5, 1 - 1e-5)
    x = mu_k + torch.exp(log_s_k) * temperature * (torch.log(u) - torch.log1p(-u))

    # Cuantización a 16 bits en [-1, 1]
    x = torch.clamp(x, -1.0, 1.0)
    x = torch.round(x * (LEVELS - 1) / 2 + (LEVELS - 1) / 2) / ((LEVELS - 1) / 2) - 1.0
    return x
