# hor

```bash
python pretrain.py --input assets/audio.wav --steps 5000 --device cuda --out ckpt.pt
python main.py --input assets/audio.wav --mode batched --step 5 --iterations 100 --device cuda --checkpoint ckpt.pt
```
