# hor

```bash
python pretrain.py --input assets/audio.wav --steps 5000 --device cuda --out base.pt
python main.py --input assets/audio.wav --mode batched --checkpoint base.pt --save-checkpoint sesion1.pt --save-every 10 --device cuda
python generate.py --checkpoint sesion1.pt --duration 60 --temperature 0.85 --out pieza.wav
```
