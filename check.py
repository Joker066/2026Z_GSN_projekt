"""Sanity checks for your implementation:  python check.py

Run it after each part you implement; it stops at the first failure.
"""
import torch

print("1. model shapes")
from sr.models import SuperResolutionNet, count_parameters

for s in (2, 3, 4):
    y = SuperResolutionNet(scale=s, channels=16, n_blocks=2)(torch.rand(1, 3, 20, 24))
    assert y.shape == (1, 3, 20 * s, 24 * s), f"scale {s}: got {tuple(y.shape)}"
n = count_parameters(SuperResolutionNet(scale=2))
print(f"   ok, default model has {n:,} params")

print("2. metrics on synthetic data")
from sr.metrics import psnr, ssim

a = torch.randint(0, 256, (2, 3, 64, 64)) / 255.0  # 8-bit-valued images
assert torch.isinf(psnr(a, a)).all() or (psnr(a, a) > 90).all(), "identical images -> inf/very high PSNR"
assert torch.allclose(ssim(a, a).float(), torch.ones(2), atol=1e-6), "identical images -> SSIM 1"
assert psnr(a, a).shape == (2,), "psnr must return one value per image"
print("   ok")

print("3. data + bicubic baseline on Set5 (x2)")
from PIL import Image

from sr.data import SREvalDataset, SRTrainDataset, to_tensor

ds = SREvalDataset("data/Set5_HR", scale=2)
P = S = 0.0
for lr, hr, *_ in ds:
    assert hr.shape[-1] == 2 * lr.shape[-1] and hr.shape[-2] == 2 * lr.shape[-2]
    # upscale LR back with PIL bicubic as the reference method
    lr_img = Image.fromarray((lr.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    bic = to_tensor(lr_img.resize((hr.shape[-1], hr.shape[-2]), Image.BICUBIC))
    P += psnr(bic[None], hr[None], crop=2).item()
    S += ssim(bic[None], hr[None], crop=2).item()
P, S = P / len(ds), S / len(ds)
print(f"   bicubic Set5 x2: PSNR {P:.2f} dB, SSIM {S:.4f}   (expected ~33.67 / ~0.930)")
assert abs(P - 33.67) < 0.15 and abs(S - 0.930) < 0.003, "metrics or degradation differ from the standard protocol"

tr = SRTrainDataset("data/Set5_HR", scale=2, lr_patch=48, crops_per_image=4)
lr, hr = tr[7]  # index > n_files also checks the idx % n_files wrap-around
assert lr.shape == (4, 3, 48, 48) and hr.shape == (4, 3, 96, 96), (lr.shape, hr.shape)
assert not torch.equal(hr[0], hr[1]), "crops from one image should differ"
print("   ok")
print("All checks passed.")
