"""
Baseline super-resolution model: small residual CNN + PixelShuffle.
Residual block: conv3x3 -> ReLU -> conv3x3, then add the input back.
Super-resolution model: head conv3x3 -> n residual blocks -> conv3x3 (output = 3 * scale ** 2) -> PixelShuffle.]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """conv3x3 -> ReLU -> conv3x3, then add the input back."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding='same')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(F.relu(self.conv1(x)))
        return x + y


class SuperResolutionNet(nn.Module):
    def __init__(self, scale: int = 2, channels: int = 64, n_blocks: int = 16):
        super().__init__()
        self.scale = scale
        self.head = nn.Conv2d(3, channels, kernel_size=3, padding='same')
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])
        self.to_rgb = nn.Conv2d(channels, 3 * self.scale ** 2, kernel_size=3, padding='same')

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        x = self.head(lr)
        for block in self.blocks:
            x = block(x)
        x = self.to_rgb(x)
        return F.pixel_shuffle(x, upscale_factor=self.scale)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = SuperResolutionNet(scale=2)
    x = torch.rand(2, 3, 48, 48)
    y = m(x)
    assert y.shape == (2, 3, 96, 96), y.shape
    print("OK", y.shape, f"{count_parameters(m):,} params")
