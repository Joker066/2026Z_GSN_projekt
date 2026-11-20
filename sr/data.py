"""Datasets that turn HR images into (LR, HR) pairs.

Degradation model used by most SR papers: LR = bicubic downscale of HR.
HR images live in data/DIV2K_train_HR, data/Set5_HR, ... (see scripts/download_data.py).
"""
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def list_images(folder) -> list[Path]:
    return sorted(Path(folder).glob("*.[pP][nN][gG]"))


def to_tensor(img: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)


def bicubic_down(img: Image.Image, scale: int) -> Image.Image:
    return img.resize((img.width // scale, img.height // scale), Image.BICUBIC)


def augment(lr_patch: Image.Image, hr_patch: Image.Image):
    """Same random flips / transpose for LR and HR."""
    if np.random.rand() > 0.5:
        hr_patch = hr_patch.transpose(Image.FLIP_LEFT_RIGHT)
        lr_patch = lr_patch.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() > 0.5:
        hr_patch = hr_patch.transpose(Image.FLIP_TOP_BOTTOM)
        lr_patch = lr_patch.transpose(Image.FLIP_TOP_BOTTOM)
    if np.random.rand() > 0.5:
        hr_patch = hr_patch.transpose(Image.TRANSPOSE)
        lr_patch = lr_patch.transpose(Image.TRANSPOSE)
    return lr_patch, hr_patch


class SRTrainDataset(Dataset):
    """Random patches from HR training images.

    Decoding a 2K PNG (~180 ms) costs far more than cropping (~0.1 ms), so each
    decoded image yields `crops_per_image` patches.

    Each item: (lr [k,3,p,p], hr [k,3,p*s,p*s]) with k = crops_per_image.
    A DataLoader batch is then [B,k,3,p,p]; flatten it in the training loop:
        lr = lr.flatten(0, 1)   # -> [B*k,3,p,p]
    """

    def __init__(self, hr_dir, scale: int = 2, lr_patch: int = 48, repeat: int = 20,
                 crops_per_image: int = 8):
        self.hr_files = list_images(hr_dir)
        self.scale = scale
        self.lr_patch = lr_patch
        self.repeat = repeat
        self.crops_per_image = crops_per_image

    def __len__(self):
        return len(self.hr_files) * self.repeat

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx % len(self.hr_files)]).convert("RGB")
        hr_width, hr_height = hr_img.size
        hr_patch_size = self.lr_patch * self.scale
        lr_patches, hr_patches = [], []
        for _ in range(self.crops_per_image):
            # Random crop coordinates for HR patch
            x = np.random.randint(0, hr_width - hr_patch_size + 1)
            y = np.random.randint(0, hr_height - hr_patch_size + 1)
            hr_patch = hr_img.crop((x, y, x + hr_patch_size, y + hr_patch_size))
            lr_patch = bicubic_down(hr_patch, self.scale)
            lr_patch, hr_patch = augment(lr_patch, hr_patch)  # independent per patch
            lr_patches.append(to_tensor(lr_patch))
            hr_patches.append(to_tensor(hr_patch))
        return torch.stack(lr_patches), torch.stack(hr_patches)


class SREvalDataset(Dataset):
    """Whole images for testing.

    Each item: (lr [3,h,w], hr [3,s*h,s*w], name).
    """

    def __init__(self, hr_dir, scale: int = 2):
        self.hr_files = list_images(hr_dir)
        self.scale = scale

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = Image.open(self.hr_files[idx]).convert("RGB")
        width, height = hr_img.size
        new_width = width - (width % self.scale)
        new_height = height - (height % self.scale)
        hr_cropped = hr_img.crop((0, 0, new_width, new_height))
        lr = bicubic_down(hr_cropped, self.scale)
        return to_tensor(lr), to_tensor(hr_cropped), self.hr_files[idx].stem
