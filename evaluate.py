"""Evaluate trained models:  python evaluate.py runs/baseline_x2/best.pt [more.pt ...]

The ablation needs:
  * a table: model | #params | PSNR | SSIM, on Set5, Set14 (and DIV2K valid),
    including Bicubic as the reference row,
  * image crops: the same zoomed region from HR / Bicubic / each model.

Outputs (in --out, default results/):
  summary.md   - mean PSNR/SSIM per set and model (paste into the report)
  metrics.csv  - per-image values
  crops/<set>/<image>.png - HR with the zoom box, then the zoomed region per method
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from sr.data import SREvalDataset, to_tensor
from sr.metrics import psnr, ssim
from sr.models import SuperResolutionNet, count_parameters


def load_checkpoint(path, device):
    """Rebuild the model from the saved hyper-parameters and load its weights."""
    checkpoint = torch.load(path, map_location=device)
    args = checkpoint["args"]
    model = SuperResolutionNet(scale=args["scale"], channels=args["channels"], n_blocks=args["n_blocks"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return args["name"], model, args["scale"]


def bicubic_up(lr: torch.Tensor, scale: int) -> torch.Tensor:
    """Reference method: upscale LR back with PIL bicubic (same resampler as the degradation)."""
    img = Image.fromarray((lr.permute(1, 2, 0).numpy() * 255).round().astype("uint8"))
    return to_tensor(img.resize((img.width * scale, img.height * scale), Image.BICUBIC))


def most_detailed_box(hr: torch.Tensor, size: int) -> tuple[int, int]:
    """Top-left (y, x) of the size x size window with the most edges (sum of gradient magnitudes)."""
    gray = hr.mean(0, keepdim=True)[None]  # [1,1,H,W]
    grad = (gray[..., 1:, :-1] - gray[..., :-1, :-1]).abs() + (gray[..., :-1, 1:] - gray[..., :-1, :-1]).abs()
    stride = max(1, size // 4)
    energy = F.avg_pool2d(grad, size, stride=stride)[0, 0]
    idx = energy.flatten().argmax().item()
    return (idx // energy.shape[1]) * stride, (idx % energy.shape[1]) * stride


@torch.no_grad()
def evaluate_set(models, dataset, device, crop_dir: Path | None, crop_size: int):
    """Per-image PSNR/SSIM for bicubic and every model; optionally save crop figures."""
    scale = dataset.scale
    rows = []
    for lr, hr, name in dataset:
        outputs = {"Bicubic": bicubic_up(lr, scale).to(device)}
        for model_name, model, _ in models:
            outputs[model_name] = model(lr.unsqueeze(0).to(device)).clamp(0, 1)[0]

        hr_b = hr.unsqueeze(0).to(device)
        scores = {}
        for method, sr in outputs.items():
            p = psnr(sr.unsqueeze(0), hr_b, crop=scale).item()
            s = ssim(sr.unsqueeze(0), hr_b, crop=scale).item()
            scores[method] = p
            rows.append({"image": name, "method": method, "psnr": p, "ssim": s})

        if crop_dir is not None and min(hr.shape[-2:]) >= crop_size:
            outputs = {m: img.cpu() for m, img in outputs.items()}
            save_crop_figure(hr, outputs, scores, name, crop_dir, crop_size)
    return rows


def save_crop_figure(hr, outputs, scores, name, crop_dir: Path, size: int):
    y, x = most_detailed_box(hr, size)
    panels = [("HR", hr, None)] + [(m, img, scores[m]) for m, img in outputs.items()]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(2.8 * (len(panels) + 1), 3.2))
    axes[0].imshow(hr.permute(1, 2, 0).numpy())
    axes[0].add_patch(plt.Rectangle((x, y), size, size, fill=False, edgecolor="red", linewidth=1.5))
    axes[0].set_title(name, fontsize=9)
    for ax, (title, img, p) in zip(axes[1:], panels):
        ax.imshow(img[:, y:y + size, x:x + size].permute(1, 2, 0).numpy(), interpolation="nearest")
        ax.set_title(title if p is None else f"{title}\n{p:.2f} dB", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    crop_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(crop_dir / f"{name}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--sets", nargs="+", default=["data/Set5_HR", "data/Set14_HR", "data/DIV2K_valid_HR"])
    p.add_argument("--out", default="results")
    p.add_argument("--crop-sets", nargs="+", default=["data/Set5_HR", "data/Set14_HR"],
                   help="sets to save crop figures for (DIV2K valid would be 100 figures)")
    p.add_argument("--crop-size", type=int, default=96, help="zoom window size in HR pixels")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_checkpoint(c, device) for c in args.checkpoints]  # load once, reuse for every set
    scales = {scale for _, _, scale in models}
    assert len(scales) == 1, f"all checkpoints must have the same scale, got {scales}"
    scale = scales.pop()
    params = {name: count_parameters(m) for name, m, _ in models}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows, summary = [], []
    for set_dir in args.sets:
        if not Path(set_dir).is_dir():
            print(f"skipping {set_dir} (not found)")
            continue
        set_name = Path(set_dir).name.removesuffix("_HR")
        dataset = SREvalDataset(set_dir, scale=scale)
        crop_dir = out / "crops" / set_name if set_dir in args.crop_sets else None
        rows = evaluate_set(models, dataset, device, crop_dir, args.crop_size)
        for r in rows:
            all_rows.append({"set": set_name, **r})
        for method in ["Bicubic"] + [name for name, _, _ in models]:
            mr = [r for r in rows if r["method"] == method]
            mean_psnr = sum(r["psnr"] for r in mr) / len(mr)
            mean_ssim = sum(r["ssim"] for r in mr) / len(mr)
            summary.append((set_name, method, params.get(method, "-"), mean_psnr, mean_ssim))
            print(f"{set_name:<12} {method:<16} PSNR {mean_psnr:.3f}  SSIM {mean_ssim:.4f}")

    with open(out / "metrics.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["set", "image", "method", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [f"Scale x{scale}. PSNR/SSIM on Y channel, {scale} px border cropped.", "",
             "| Set | Method | Params | PSNR [dB] | SSIM |", "|---|---|---|---|---|"]
    for set_name, method, n, mp, ms in summary:
        n = f"{n:,}" if isinstance(n, int) else n
        lines.append(f"| {set_name} | {method} | {n} | {mp:.3f} | {ms:.4f} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {out / 'summary.md'}, {out / 'metrics.csv'}, crops in {out / 'crops'}")


if __name__ == "__main__":
    main()
