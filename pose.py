#!/usr/bin/env python3
'''
Deteccion de una persona con MediaPipe Pose (Tasks API) para generar audio espacial.

Pipeline:
    1. Captura de video con OpenCV.
    2. Deteccion de pose con PoseLandmarker (modo VIDEO, sincrono).
    3. Proximidad relativa: distancia en pixeles entre hombros, normalizada por el
       ancho del frame. No se usa el landmark z porque no esta calibrado a unidades
       metricas y es ruidoso.
    4. Posicion horizontal en pantalla: promedio en x de hombros y caderas, mapeado
       a [-1, 1].
    5. Suavizado con media movil exponencial (EMA) sobre ambas magnitudes.
    6. Sintesis de audio en tiempo real (tono senoidal, paneo por ley del seno y
       coseno, frecuencia y ganancia derivadas de la proximidad).
    7. Envio simultaneo por OSC de /pose/proximity y /pose/position.

Dependencias: opencv-python, mediapipe, numpy, sounddevice, python-osc.
En Debian 13, sounddevice requiere la libreria nativa PortAudio:
    sudo apt install libportaudio2
'''

import argparse
import functools
import math
import os
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import sounddevice as sd

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/' 'pose_landmarker_lite/float16/1/pose_landmarker_lite.task'

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

SAMPLE_RATE = 44100
BLOCK_SIZE = 1024


class Ema:
    '''Media movil exponencial de un escalar.'''

    def __init__(self, alpha):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value


class AudioState:
    '''
    Estado compartido entre el hilo principal (captura y deteccion) y el hilo de
    audio (callback de sounddevice).

    proximity y position se leen y escriben desde ambos hilos, por lo que estan
    protegidos por un lock. phase, prev_freq y prev_gain solo los toca el callback
    de audio, asi que no requieren proteccion adicional.
    '''

    def __init__(self, sample_rate, min_freq, max_freq, max_gain):
        self.lock = threading.Lock()
        self.sample_rate = sample_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_gain = max_gain
        self.proximity = 0.0
        self.position = 0.0
        self.phase = 0.0
        self.prev_freq = min_freq
        self.prev_gain = 0.0

    def set_pose(self, proximity, position):
        with self.lock:
            self.proximity = proximity
            self.position = position

    def get_pose(self):
        with self.lock:
            return self.proximity, self.position


def audio_callback(outdata, frames, time_info, status, state):
    if status:
        print(status, file=sys.stderr)

    proximity, position = state.get_pose()

    freq_target = state.min_freq + proximity * (state.max_freq - state.min_freq)
    gain_target = proximity * state.max_gain

    # Interpolacion lineal de frecuencia y ganancia dentro del bloque para evitar
    # discontinuidades audibles (clicks) al cambiar de un bloque a otro.
    freqs = np.linspace(state.prev_freq, freq_target, frames)
    gains = np.linspace(state.prev_gain, gain_target, frames)

    phase_increments = 2.0 * np.pi * freqs / state.sample_rate
    phases = state.phase + np.cumsum(phase_increments)
    wave = np.sin(phases) * gains

    # Paneo por ley del seno y coseno: mantiene energia constante entre canales.
    pan = min(max((position + 1.0) / 2.0, 0.0), 1.0)
    left_gain = math.cos(pan * math.pi / 2.0)
    right_gain = math.sin(pan * math.pi / 2.0)

    outdata[:, 0] = wave * left_gain
    outdata[:, 1] = wave * right_gain

    state.phase = float(phases[-1] % (2.0 * np.pi))
    state.prev_freq = freq_target
    state.prev_gain = gain_target


def ensure_model_downloaded(model_path):
    if os.path.exists(model_path):
        return
    directory = os.path.dirname(os.path.abspath(model_path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    print(f'Descargando modelo PoseLandmarker en {model_path} ...')
    urllib.request.urlretrieve(MODEL_URL, model_path)


def shoulder_fraction(landmarks, frame_w, frame_h):
    '''Distancia entre hombros en pixeles, normalizada por el ancho del frame.'''
    left = landmarks[LEFT_SHOULDER]
    right = landmarks[RIGHT_SHOULDER]
    x1, y1 = left.x * frame_w, left.y * frame_h
    x2, y2 = right.x * frame_w, right.y * frame_h
    return math.hypot(x2 - x1, y2 - y1) / frame_w


def draw_debug(frame, pose_landmarks, proximity, position):
    h, w = frame.shape[:2]
    if pose_landmarks:
        for lm in pose_landmarks[0]:
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
    cv2.putText(frame, f'proximidad: {proximity:.2f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'posicion: {position:+.2f}', (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    bar_x = int((position + 1.0) / 2.0 * w)
    cv2.line(frame, (bar_x, h - 10), (bar_x, h - 30), (0, 200, 255), 3)


def parse_args():
    parser = argparse.ArgumentParser(description='Pose a audio espacial con MediaPipe.')
    parser.add_argument('--camera-index', type=int, default=0)
    parser.add_argument('--model-path', type=str, default='pose_landmarker_lite.task')

    parser.add_argument('--mirror', dest='mirror', action='store_true', default=True)
    parser.add_argument('--no-mirror', dest='mirror', action='store_false')

    parser.add_argument('--show', dest='show', action='store_true', default=True)
    parser.add_argument('--no-show', dest='show', action='store_false')

    parser.add_argument('--smoothing', type=float, default=0.25, help='alpha de la EMA, entre 0 (muy suave) y 1 (sin suavizado)')

    parser.add_argument('--min-shoulder-frac', type=float, default=0.08, help='fraccion del ancho de frame considerada "lejos"')
    parser.add_argument('--max-shoulder-frac', type=float, default=0.45, help='fraccion del ancho de frame considerada "cerca"')

    parser.add_argument('--min-freq', type=float, default=220.0)
    parser.add_argument('--max-freq', type=float, default=880.0)
    parser.add_argument('--max-gain', type=float, default=0.6)

    parser.add_argument('--osc', dest='osc', action='store_true', default=True)
    parser.add_argument('--no-osc', dest='osc', action='store_false')
    parser.add_argument('--osc-ip', type=str, default='127.0.0.1')
    parser.add_argument('--osc-port', type=int, default=9000)

    return parser.parse_args()


def main():
    args = parse_args()
    ensure_model_downloaded(args.model_path)

    osc_client = None
    if args.osc:
        from pythonosc.udp_client import SimpleUDPClient

        osc_client = SimpleUDPClient(args.osc_ip, args.osc_port)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError('No se pudo abrir la camara indicada.')

    base_options = mp_python.BaseOptions(model_asset_path=args.model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    audio_state = AudioState(SAMPLE_RATE, args.min_freq, args.max_freq, args.max_gain)
    stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=2,
        dtype='float32',
        blocksize=BLOCK_SIZE,
        callback=functools.partial(audio_callback, state=audio_state),
    )

    proximity_ema = Ema(args.smoothing)
    position_ema = Ema(args.smoothing)
    last_position = 0.0
    start_time = time.time()

    try:
        with stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                h, w = frame.shape[:2]

                if result.pose_landmarks:
                    lm = result.pose_landmarks[0]
                    raw_fraction = shoulder_fraction(lm, w, h)
                    span = args.max_shoulder_frac - args.min_shoulder_frac
                    proximity_target = min(max((raw_fraction - args.min_shoulder_frac) / span, 0.0), 1.0)

                    cx = (lm[LEFT_SHOULDER].x + lm[RIGHT_SHOULDER].x + lm[LEFT_HIP].x + lm[RIGHT_HIP].x) / 4.0
                    position_target = cx * 2.0 - 1.0
                    last_position = position_target
                else:
                    proximity_target = 0.0
                    position_target = last_position

                proximity_smooth = proximity_ema.update(proximity_target)
                position_smooth = position_ema.update(position_target)
                audio_state.set_pose(proximity_smooth, position_smooth)

                if osc_client is not None:
                    osc_client.send_message('/pose/proximity', float(proximity_smooth))
                    osc_client.send_message('/pose/position', float(position_smooth))

                if args.show:
                    draw_debug(frame, result.pose_landmarks, proximity_smooth, position_smooth)
                    cv2.imshow('pose-audio', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == '__main__':
    main()
