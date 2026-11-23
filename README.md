# Super-resolution with channel attention

Assignment: `instructions.pdf`. Plan:
1. **Baseline:** residual CNN + PixelShuffle (×2). ✅ done
2. **Channel attention** inside the residual blocks. ⏳ next
3. **Ablation:** baseline vs. baseline + CA at a similar parameter count, compared with PSNR/SSIM and image crops.

## Layout

```
sr/models.py        ResidualBlock, SuperResolutionNet, count_parameters
sr/data.py          SRTrainDataset (random patches), SREvalDataset (whole images)
sr/metrics.py       PSNR / SSIM (Y channel, border crop = scale, as in SR papers)
train.py            training loop -> runs/<name>/{best.pt, last.pt, log.csv}
evaluate.py         table + crop figures -> results/
check.py            sanity checks (shapes, metrics, bicubic reference)
scripts/download_data.py
```

## Setup

```bash
pip install -r requirements.txt   # torch with CUDA
python scripts/download_data.py   # DIV2K train/valid + Set5 + Set14 into data/ (~3.8 GB)
python check.py                   # should end with "All checks passed."
```

`data/`, `runs/`, `results/` and `*.pt` are git-ignored. I can send you my `runs/baseline_x2/best.pt`, so you don't have to retrain the baseline.

## Train

```bash
# smoke test, about 2 min: val PSNR on Set5 should get above bicubic (33.67 dB)
python train.py --name smoke --iterations 2000 --val-every 500

# full baseline (100k iterations at default, about 1.5h)
python train.py --name baseline_x2

# continue an interrupted run
python train.py --name baseline_x2 --resume
```

Defaults: ×2, 64 channels, 16 blocks, L1 loss, Adam with lr 2e-4 (halved at 50% and 75% of `--iterations`), batch of 16 patches (48×48 LR), 8 crops per decoded image, validation on Set5 every 1000 iterations.

## Evaluate

```bash
python evaluate.py runs/baseline_x2/best.pt                       # one model
python evaluate.py runs/baseline_x2/best.pt runs/ca_x2/best.pt    # ablation
```

This writes three outputs to `results/`:
- `summary.md`: mean PSNR/SSIM per set, with Bicubic as the reference row.
- `metrics.csv`: values per image.
- `crops/<set>/*.png`: zoomed comparisons of the region with the most edges.

Evaluating Set5, Set14 and DIV2K valid takes about 4 min.

## Baseline results (×2, 100k iterations)

| Set | Method | Params | PSNR [dB] | SSIM |
|---|---|---|---|---|
| Set5 | Bicubic | – | 33.673 | 0.9303 |
| Set5 | baseline_x2 | 1,190,412 | 37.717 | 0.9596 |
| Set14 | Bicubic | – | 30.325 | 0.8698 |
| Set14 | baseline_x2 | 1,190,412 | 33.296 | 0.9152 |
| DIV2K valid | Bicubic | – | 32.435 | 0.9038 |
| DIV2K valid | baseline_x2 | 1,190,412 | 35.766 | 0.9429 |

## Next: channel attention

The PDF (slide 2) sketches the module: `AdaptiveAvgPool2d(1)` → 1×1 conv (C → C/r) → ReLU → 1×1 conv (C/r → C) → Sigmoid, then multiply it with the input.

I think this part will change these files:
- **`sr/models.py`:** `ResidualBlock.forward` computes `y = conv2(relu(conv1(x)))` and returns `x + y`. The slide applies attention to `y` just before the addition. Some kind of on/off switch on `ResidualBlock` / `SuperResolutionNet` would let one model class cover both sides of the ablation.
- **`train.py`:** builds the model from CLI args and saves them in `checkpoint["args"]`, so a new flag there gets stored automatically.
- **`evaluate.py`:** `load_checkpoint` rebuilds the model from those saved args. It will probably need to read the new flag too, with a default so the existing baseline checkpoint still loads.
- **`check.py` + the smoke test** were handy for catching mistakes before a long run.

Things I think matter for a fair comparison:
- Both models trained with the same iterations, schedule and data, so the only difference is the attention module.
- Parameter counts in the table. By my estimate, CA with r=8 adds about 1k parameters per block (~17k total, +1.5%), which is pretty close.
- If shorter runs are easier (e.g. `--iterations 50000`), we can retrain the baseline at the same budget so the numbers stay comparable.
