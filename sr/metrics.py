"""PSNR / SSIM, using the standard SR-paper protocol:
  * convert RGB to the Y channel (luminance) of YCbCr, ITU-R BT.601,
  * crop `scale` pixels from every border,
  * images in [0,1], clamp and round SR output to 8-bit levels first.

Sanity check (python check.py): bicubic x2 on Set5 should give
about 33.6-33.7 dB PSNR and about 0.930 SSIM.
"""
import torch
import torch.nn.functional as F

def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    R_channel = img[:, 0:1, :, :]
    G_channel = img[:, 1:2, :, :]
    B_channel = img[:, 2:3, :, :]
    Y_channel = 16 + (65.481 * R_channel + 128.553 * G_channel + 24.966 * B_channel)
    return Y_channel / 255.0


def psnr(sr: torch.Tensor, hr: torch.Tensor, crop: int = 0) -> torch.Tensor:
    """Per-image PSNR in dB, shape [B]."""
    if crop > 0:
        sr = sr[:, :, crop:-crop, crop:-crop]
        hr = hr[:, :, crop:-crop, crop:-crop]
    y_sr = rgb_to_y(sr)
    y_hr = rgb_to_y(hr)
    
    return 10 * torch.log10(1 / torch.mean((y_sr - y_hr) ** 2, dim=(1, 2, 3)))


def ssim(sr: torch.Tensor, hr: torch.Tensor, crop: int = 0) -> torch.Tensor:
    """Per-image SSIM, shape [B]. Wang et al. 2004."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    gaussian_1D = torch.arange(-5, 6, dtype=torch.float64, device=sr.device)
    gaussian_1D = torch.exp(-0.5 * (gaussian_1D / 1.5) ** 2)
    gaussian_1D /= gaussian_1D.sum()
    gaussian_2D = torch.outer(gaussian_1D, gaussian_1D)
    gaussian_2D = torch.reshape(gaussian_2D, (1, 1, 11, 11)).to(dtype=torch.float64)

    if crop > 0:
        sr = sr[:, :, crop:-crop, crop:-crop]
        hr = hr[:, :, crop:-crop, crop:-crop]
    
    y_sr = rgb_to_y(sr).to(dtype=torch.float64)
    y_hr = rgb_to_y(hr).to(dtype=torch.float64)

    mu_x = F.conv2d(y_sr, gaussian_2D)
    mu_y = F.conv2d(y_hr, gaussian_2D)
    sigma_x = F.conv2d(y_sr * y_sr, gaussian_2D) - mu_x ** 2
    sigma_y = F.conv2d(y_hr * y_hr, gaussian_2D) - mu_y ** 2
    sigma_xy = F.conv2d(y_sr * y_hr, gaussian_2D) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    
    return ssim_map.mean(dim=(1, 2, 3))
