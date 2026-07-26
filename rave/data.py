'''Dataset de audio con segmentación deslizante.

Carga uno o más archivos de audio, los resamplea a la tasa objetivo, los
normaliza al rango [-1, 1] y extrae segmentos de longitud fija con hop
configurable (por defecto 50% de solape).

Segmentos y split:
    * Índice global de segmentos: lista de tuplas (path_idx, start_sample).
    * Split train/val por asignación aleatoria de índices con semilla fija
      (comparable entre corridas). Para archivo único, el "val" es material
      del mismo corpus, por lo que sirve como early-stopping suave pero no
      como test de generalización.
'''

from __future__ import annotations

from pathlib import Path
from torch.utils.data import Dataset

import random
import soundfile as sf
import torch
import torch.nn.functional as F


def _load_audio_file(path: Path) -> tuple[torch.Tensor, int]:
    '''Carga un archivo con soundfile y devuelve (waveform, sample_rate).

    waveform tiene forma (channels, samples).
    '''
    data, sr = sf.read(str(path), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data.T).contiguous()
    return waveform, int(sr)


def _resample(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    '''Resamplea via interpolación linear en la salida FFT.

    Implementación minimalista: se hace en torch usando la fórmula de
    conversión por relación racional. Para calidad crítica sustituir por
    torchaudio.functional.resample o resampy.
    '''
    if orig_sr == target_sr:
        return waveform
    n_out = int(waveform.size(-1) * target_sr / orig_sr)
    return F.interpolate(waveform.unsqueeze(0), size=n_out, mode='linear', align_corners=False).squeeze(0)


class AudioSegmentDataset(Dataset):
    '''Dataset de segmentos de audio con carga en memoria.

    Args:
        audio_paths: lista de paths a archivos de audio.
        segment_length: longitud del segmento en muestras a la tasa objetivo.
            Debe ser múltiplo de total_stride del modelo.
        hop_length: paso entre segmentos consecutivos, en muestras. Para
            solape del 50% usar segment_length // 2.
        sample_rate: tasa objetivo en Hz. Los archivos con tasa distinta se
            resamplean al cargarlos.
        normalize: si True, cada archivo se divide por el máximo absoluto de
            su forma de onda al cargarlo.
    '''

    def __init__(self, audio_paths: list[str | Path], segment_length: int = 32_768, hop_length: int | None = None, sample_rate: int = 16_000, normalize: bool = True):
        super().__init__()
        if hop_length is None:
            hop_length = segment_length // 2
        self.segment_length = int(segment_length)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)

        self.audios: list[torch.Tensor] = []
        self.paths: list[Path] = []
        for p in audio_paths:
            path = Path(p)
            waveform, sr = _load_audio_file(path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sample_rate:
                waveform = _resample(waveform, sr, sample_rate)
            waveform = waveform.squeeze(0)
            if normalize:
                peak = waveform.abs().max().clamp(min=1e-6)
                waveform = waveform / peak
            self.audios.append(waveform.contiguous())
            self.paths.append(path)

        self.segments: list[tuple[int, int]] = []
        for idx, audio in enumerate(self.audios):
            if audio.numel() < self.segment_length:
                continue
            n_starts = 1 + (audio.numel() - self.segment_length) // self.hop_length
            for i in range(n_starts):
                start = i * self.hop_length
                self.segments.append((idx, start))

        if not self.segments:
            raise ValueError(f'ningún segmento extraído: los archivos son más cortos que ' f'{segment_length} muestras a {sample_rate} Hz')

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> torch.Tensor:
        audio_idx, start = self.segments[idx]
        segment = self.audios[audio_idx][start : start + self.segment_length]
        return segment.unsqueeze(0)  # (1, T)


def split_train_val(dataset: AudioSegmentDataset, val_fraction: float = 0.1, seed: int = 42) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    '''Split train/val determinista sobre índices de segmento.'''
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = max(1, int(n * val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    return train_set, val_set
