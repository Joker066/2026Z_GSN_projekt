# P1 cheat sheet: baseline SR ×2 (residual CNN + PixelShuffle)

> Citations in square brackets, e.g. [EDSR], refer to the [References](#references) section at the end.
> Paper numbers and hyper-parameters marked *(verify)* were written from memory, so check them in the paper before quoting them in the report.

## The problem in one sentence

Given a low-resolution (LR) image, reconstruct a 2× bigger high-resolution (HR) one.
It's supervised: **LR = bicubic ×2 downscale of HR**, the standard "BI" degradation [DIV2K, EDSR, RDN, RCAN].
This gives perfectly aligned pairs and unlimited data, and train and test use the same degradation.

**Data:** train on **DIV2K train** (800 2K images) [DIV2K]. Validate during training on **Set5** [Set5].
Test on **Set5, Set14** [Set14] **and DIV2K valid**. Only HR files are on disk; LR is generated on the fly.

---

## 1. Inputs and outputs

- **Input:** LR **RGB** image, float in **[0,1]**, shape `[B,3,H,W]`. It is **not** upscaled beforehand (unlike [SRCNN], which bicubic-upscales the input first) and not normalized.
- **Output:** SR image `[B,3,2H,2W]`, compared with the original HR.
- **Fully convolutional**, so it accepts any size: 48×48 patches in training, whole images in evaluation.

## 2. Tensor shapes (training batch)

```
DataLoader      [2, 8, 3, 48, 48]   2 images × 8 random crops
flatten(0,1)    [16, 3, 48, 48]
head conv3×3    [16, 64, 48, 48]
16× ResBlock    [16, 64, 48, 48]    padding='same', so H×W unchanged
to_rgb conv3×3  [16, 12, 48, 48]    12 = 3 colours × 2² sub-pixels
PixelShuffle    [16,  3, 96, 96]    = HR target shape
```

### What does `flatten(0, 1)` do, and why is it needed?

It **merges dimensions 0 and 1 into one**: `[2, 8, 3, 48, 48]` (B, k, C, H, W) → `[16, 3, 48, 48]` (B·k, C, H, W).

- **Why the extra dimension exists:** one dataset item is **8 crops from the same image** (`[8,3,48,48]`), because decoding a 2K PNG (~180 ms) is much slower than cropping (~0.1 ms). The DataLoader stacks 2 items into a batch, which adds a dimension, so the tensor is 5D.
- **Why it has to go:** `Conv2d` expects **4D** input `[N, C, H, W]`. After flattening, the model sees 16 independent patches. Nothing is copied or changed; the same numbers are just viewed with a different shape. `lr` and `hr` are flattened the same way, so patch *i* still matches target *i*.

## 3. Main layers

- **head:** RGB → 64 feature channels.
- **ResidualBlock** [ResNet, EDSR]: `y = conv(ReLU(conv(x)))`, `return x + y`.
  - It learns a **correction** to `x`. If there is nothing useful to add, it can output y ≈ 0.
  - The skip makes the gradient `∂(x+y)/∂x = 1 + ∂y/∂x`, so 32 stacked convs stay trainable.
  - **No BatchNorm** [EDSR]: BN removes absolute brightness/contrast info that SR needs, and it costs memory.
- **to_rgb:** for each LR pixel, predicts **4 sub-pixels × 3 colours = 12 values**. No activation, so the output is linear (no clipping).
- **PixelShuffle** [ESPCN]: rearranges channels into space, with **no weights**. All computation runs at LR, which is about 4× cheaper than upscaling first [SRCNN, VDSR]. There are no checkerboard artifacts (unlike transposed conv).
- **Receptive field:** 34 convs 3×3 → 1 + 2·34 = **69×69 LR pixels**.
- **Size:** 16 blocks × 64 channels = the **EDSR baseline** configuration [EDSR]. Both are CLI flags (`--n_blocks`, `--channels`). Parameters grow with C² and linearly with depth.

### PixelShuffle: the exact formula

"Sub-pixels" are the 2×2 HR pixels that fall inside the area of one LR pixel.

```
input  [B, C·r², H, W]   →   output [B, C, H·r, W·r]

out[b, c, h·r + i, w·r + j] = in[b, c·r² + i·r + j, h, w],   i, j ∈ {0, …, r−1}
```

For r = 2, C = 3: channels 0–3 are red, 4–7 green, 8–11 blue. Within each colour, for an LR pixel `(h, w)`:

| channel | HR position |
|---|---|
| c·4 + 0 | (2h, 2w), top-left |
| c·4 + 1 | (2h, 2w+1), top-right |
| c·4 + 2 | (2h+1, 2w), bottom-left |
| c·4 + 3 | (2h+1, 2w+1), bottom-right |

In code it's just a reshape + permute, with no arithmetic:

```python
x.view(B, C, r, r, H, W).permute(0, 1, 4, 2, 5, 3).reshape(B, C, H*r, W*r)
#        b  c  i  j  h  w             b  c  h  i  w  j
```

The mapping is **fixed**. The *learning* happens in `to_rgb`: the gradient flows straight through PixelShuffle, so `to_rgb` learns which value to put in which channel.

## 4. Loss: L1

`mean |SR − HR|` over all pixels and channels [EDSR].

- **Sharper than MSE.** MSE averages plausible solutions, which gives blur. L1 is also more robust to outliers.
- [EDSR] reports that L1 trains better than L2. [Zhao et al.] study this in detail.
- "But PSNR is MSE-based?" Yes, but in practice L1 still gives better PSNR.

### Should we consider other losses?

| Loss | Effect |
|---|---|
| **MSE (L2)** | Directly optimizes PSNR, but in practice blurrier and slower to converge than L1 [Zhao et al.] |
| **Charbonnier** (smooth L1) [LapSRN] | Like L1 but differentiable at 0. Often equal or slightly better |
| **L1 + (MS-)SSIM** [Zhao et al.] | Better structure / SSIM |
| **Perceptual (VGG) + adversarial (GAN)** [SRGAN] | Much sharper, more realistic textures, but **lower PSNR/SSIM** because they invent plausible detail |

**Decision: keep L1.** It's the standard for PSNR-oriented SR [EDSR, RCAN], we're evaluated on PSNR/SSIM, and the P2 ablation is about channel attention; a different loss would add a second variable.

## 5. Trained parameters: all **1,190,412**

| Layer | Calculation | Params |
|---|---|---|
| head | 3·64·9 + 64 | 1,792 |
| 16 blocks | 32 × (64·64·9 + 64) | 1,181,696 (99.3%) |
| to_rgb | 64·12·9 + 12 | 6,924 |
| ReLU, PixelShuffle | – | 0 |
| **Total** | weights 1,188,288 + biases 2,124 | **1,190,412** |

The weights and biases of **34 convs**. Nothing is frozen, nothing is pretrained, and there are no BN buffers.
Conv formula: `in · out · k·k + out`.

### Training recipe and where it came from

| Setting | Our value | Source |
|---|---|---|
| Architecture (residual blocks without BN, 16 × 64) | same | [EDSR], EDSR baseline |
| Post-upsampling with PixelShuffle | same | [ESPCN], [EDSR] |
| Loss | L1 | [EDSR] |
| Optimizer | Adam | [Adam], used by [EDSR] |
| Adam betas | (0.9, 0.99) | [BasicSR] configs. The EDSR paper uses β₂ = 0.999 *(verify)* |
| Learning rate | 2e-4 | [BasicSR] configs. The EDSR paper uses 1e-4 *(verify)* |
| LR schedule | MultiStepLR, ×0.5 at 50% and 75% of training | Step decay as in [EDSR] (halved every 2·10⁵ updates *(verify)*), milestones scaled to our budget |
| Patches | 48×48 LR / 96×96 HR | [EDSR] |
| Batch | 16 patches | [EDSR] |
| Augmentation | random flips + transpose (8 orientations) | [EDSR] (flips + 90° rotations) |
| Training data | DIV2K train, bicubic ×2 | [DIV2K], [EDSR] |
| Iterations | 100k (≈ 12.5 epochs, ≈ 1.5 h) | our compute budget (EDSR trains much longer) |

### Why Adam?

> "We followed the established SR recipe: [EDSR] and most later SR papers [RCAN] use Adam. It adapts the step size per parameter, which helps in a deep network without BatchNorm, and it needs almost no tuning. It worked as expected: the smoke test beats bicubic within 2000 iterations, and the loss in `log.csv` decreases smoothly. We didn't compare it with SGD, because the optimizer isn't what our ablation is about."

- SGD was used by [SRCNN] (slow convergence) and [VDSR] (needed a high lr + gradient clipping).
- "SGD generalizes better" is known mainly from image classification. In SR (pixel regression, lots of data, little overfitting), that advantage doesn't really show up.
- L1's gradient is roughly the sign of the error (±1) and doesn't shrink as the model improves; Adam's normalization handles that smoothly.

### Why step decay and not cosine?

It follows the EDSR-style recipe. Cosine would very likely give similar results. What matters for the ablation is that **both models use the same schedule**.

## 6. Metrics: how to read them

Computed on **SR vs HR**, on the **Y channel** (luminance, ITU-R BT.601), with a **2 px border crop**, **per image, then averaged** over the set.
This is the protocol from [SRCNN]'s evaluation code, used by later papers, so results are comparable with the literature.

```
Y = (16 + 65.481·R + 128.553·G + 24.966·B) / 255
```

- **PSNR** = 10·log10(1/MSE) [dB], higher is better. +1 dB ≈ 21% lower MSE. +0.1 to 0.2 dB counts in papers.
- **SSIM** [SSIM] ∈ [0,1]: local luminance, contrast and structure (11×11 Gaussian window, σ = 1.5). Closer to perception, and 1 means identical.
- **Bicubic upscaling** is the reference row, the "no learning" minimum.
- **Why RGB in, Y for metrics?** The output has to be colour. Y is the evaluation convention, and humans are most sensitive to sharpness in luminance.

---

## Results (×2, 100k iterations)

| Set | Bicubic | Baseline (1.19M params) |
|---|---|---|
| Set5 | 33.67 / 0.9303 | **37.72 / 0.9596** |
| Set14 | 30.33 / 0.8698 | **33.30 / 0.9152** |
| DIV2K valid | 32.44 / 0.9038 | **35.77 / 0.9429** |

Comparison with the literature on Set5 ×2, PSNR / SSIM *(verify)*:

| Method | Set5 | Set14 |
|---|---|---|
| Bicubic | 33.66 / 0.9299 | 30.24 / 0.8688 |
| SRCNN [SRCNN] | 36.66 / 0.9542 | 32.45 / 0.9067 |
| VDSR [VDSR] | 37.53 / 0.9587 | 33.03 / 0.9124 |
| **Ours** | **37.72 / 0.9596** | **33.30 / 0.9152** |
| EDSR [EDSR] (43M params) | 38.11 / 0.9601 | 33.92 / 0.9195 |
| RCAN [RCAN] | 38.27 / 0.9614 | 34.12 / 0.9216 |

On Set5 this is **+4 dB over bicubic**, between VDSR and EDSR, with about 36× fewer parameters than full EDSR.
Our bicubic differs slightly from the papers' because we use PIL instead of MATLAB `imresize` (33.67 vs 33.66 on Set5, which is negligible).

---

## Grading points → what to show

| Criterion | Pts | What to do / say |
|---|---|---|
| Understanding (oral) | 4 | Sections 1–6 |
| Working baseline end-to-end | 3 | `python check.py` → "All checks passed.", then `python evaluate.py runs/baseline_x2/best.pt` → table. Have `log.csv` ready (PSNR climbs, with jumps after the lr drops) |
| Correct evaluation and metrics | 2 | Y channel + border crop, bicubic reference, 3 test sets. Crop figures from `results/crops/Set5/` |
| Runnable, organized code | 1 | `sr/` package (models, data, metrics), separate train/evaluate scripts, README with commands, CLI flags, args saved in checkpoints |

## Say these before he asks

1. **`best.pt` is selected on Set5, so Set5 is slightly optimistic. Set14 and DIV2K valid are the clean numbers.**
2. Training validation doesn't clamp the output and `evaluate.py` does, so small differences between them are expected.
3. **Bicubic degradation isn't realistic** for real photos [KernelGAN, Real-ESRGAN], but it's the standard benchmark.
4. PIL bicubic vs MATLAB `imresize`: negligible difference (see Results).
5. No global skip connection (EDSR has one). It's a cheap possible improvement.

## Likely follow-ups

- **Why PixelShuffle?** Convs keep the size, so something has to upsample. PixelShuffle does it at the end: cheap, learned, no artifacts [ESPCN].
- **Why bicubic downscaling?** It's the standard setting [DIV2K, EDSR, RCAN], and train and test must match. [SRCNN] uses bicubic **up** as input preprocessing; we don't.
- **Is bicubic necessary?** No. Any degradation would work if train and test match, but bicubic makes the results comparable. [RDN] names the settings BI (bicubic), BD (blur + downscale) and DN (noise).
- **Why 64 / 16?** EDSR baseline: small, known to work, trains in 1.5 h. Kept fixed for the ablation.
- **Next (P2):** channel attention [RCAN] on `y` before `x + y` (global avg pool → 1×1 conv C→C/8 → ReLU → 1×1 conv C/8→C → sigmoid → multiply). It adds about 17.5k parameters (+1.5%). Same training for both models.

## Useful Polish terms

funkcja straty (L1), kształty tensorów, połączenie rezydualne / skip connection, warstwa splotowa 3×3, liczba parametrów trenowalnych, degradacja (bicubic downscaling), kanał luminancji Y, przycięcie brzegu, zbiór walidacyjny vs testowy.

---

## References

- **[EDSR]** B. Lim, S. Son, H. Kim, S. Nah, K. M. Lee. *Enhanced Deep Residual Networks for Single Image Super-Resolution.* CVPR Workshops (NTIRE) 2017. arXiv:1707.02921. Code: https://github.com/sanghyun-son/EDSR-PyTorch
  Main source of the recipe: residual blocks without BN, 16 × 64 baseline, L1, Adam, 48×48 patches, DIV2K bicubic.
- **[BasicSR]** X. Wang, L. Xie, K. Yu, K. C. K. Chan, C. C. Loy, C. Dong. *BasicSR: Open Source Image and Video Restoration Toolbox.* https://github.com/XPixelGroup/BasicSR
  Source of lr 2e-4 and Adam betas (0.9, 0.99) (EDSR training configs in `options/train/EDSR/`) *(verify)*.
- **[ESPCN]** W. Shi, J. Caballero, F. Huszár, J. Totz, A. P. Aitken, R. Bishop, D. Rueckert, Z. Wang. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network.* CVPR 2016. arXiv:1609.05158
  PixelShuffle (sub-pixel convolution).
- **[DIV2K]** E. Agustsson, R. Timofte. *NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study.* CVPR Workshops 2017.
  The DIV2K dataset and the bicubic degradation track.
- **[RCAN]** Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, Y. Fu. *Image Super-Resolution Using Very Deep Residual Channel Attention Networks.* ECCV 2018. arXiv:1807.02758
  Channel attention in residual blocks (P2).
- **[RDN]** Y. Zhang, Y. Tian, Y. Kong, B. Zhong, Y. Fu. *Residual Dense Network for Image Super-Resolution.* CVPR 2018. arXiv:1802.08797
  BI / BD / DN degradation settings.
- **[SRCNN]** C. Dong, C. C. Loy, K. He, X. Tang. *Image Super-Resolution Using Deep Convolutional Networks.* TPAMI 2016 (ECCV 2014 version: *Learning a Deep Convolutional Network for Image Super-Resolution*). arXiv:1501.00092
  First CNN for SR; bicubic pre-upsampling; Y-channel evaluation convention.
- **[VDSR]** J. Kim, J. K. Lee, K. M. Lee. *Accurate Image Super-Resolution Using Very Deep Convolutional Networks.* CVPR 2016. arXiv:1511.04587
- **[ResNet]** K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385
- **[Adam]** D. P. Kingma, J. Ba. *Adam: A Method for Stochastic Optimization.* ICLR 2015. arXiv:1412.6980
- **[SSIM]** Z. Wang, A. C. Bovik, H. R. Sheikh, E. P. Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004.
- **[Zhao et al.]** H. Zhao, O. Gallo, I. Frosio, J. Kautz. *Loss Functions for Image Restoration with Neural Networks.* IEEE TCI 2017. arXiv:1511.08861
- **[LapSRN]** W.-S. Lai, J.-B. Huang, N. Ahuja, M.-H. Yang. *Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution.* CVPR 2017. arXiv:1704.03915
- **[SRGAN]** C. Ledig et al. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.* CVPR 2017. arXiv:1609.04802
- **[Set5]** M. Bevilacqua, A. Roumy, C. Guillemot, M.-L. Alberi Morel. *Low-Complexity Single-Image Super-Resolution based on Nonnegative Neighbor Embedding.* BMVC 2012.
- **[Set14]** R. Zeyde, M. Elad, M. Protter. *On Single Image Scale-Up Using Sparse-Representations.* Curves and Surfaces 2010.
- **[KernelGAN]** S. Bell-Kligler, A. Shocher, M. Irani. *Blind Super-Resolution Kernel Estimation using an Internal-GAN.* NeurIPS 2019. arXiv:1909.06581
- **[Real-ESRGAN]** X. Wang, L. Xie, C. Dong, Y. Shan. *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.* ICCV Workshops 2021. arXiv:2107.10833
